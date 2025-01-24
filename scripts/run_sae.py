import torch as t
import polars as pl
import argparse

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=["gpt2", "pythia"])
args = parser.parse_args()

device = "cuda"
langs = ["eng_Latn", "deu_Latn", "rus_Cyrl", "isl_Latn", "spa_Latn", "por_Latn", "fra_Latn", "zho_Hans", "jpn_Jpan", "kor_Hang", "hin_Deva", "arb_Arab"]

if args.model == "gpt2":
    lm_name = "gpt2-small"
    sae_release = "gpt2-small-res-jb"
    sae_ids = [f"blocks.{i}.hook_resid_pre" for i in range(12)]
    d_sae = 24576
elif args.model == "pythia":
    lm_name = "EleutherAI/pythia-70m-deduped"
    sae_release = "pythia-70m-deduped-res-sm"
    sae_ids = [f"blocks.{i}.hook_resid_post" for i in range(6)]
    d_sae = 32768
else:
    raise ValueError(f"{args.model = }")

lang_sentences = dict()

for lang in (p := tqdm(langs[1:])):
    p.set_description(lang)
    
    dataset = load_dataset(
        "facebook/flores",
        f"eng_Latn-{lang}",
        trust_remote_code=True,
        split="dev"
    )

    lang_sentences[langs[0]] = dataset[f"sentence_{langs[0]}"]
    lang_sentences[lang] = dataset[f"sentence_{lang}"]

lm = HookedTransformer.from_pretrained(lm_name, device=device)

@t.inference_mode()
def run_sae(sae, sentences):
    tokens = lm.tokenizer(sentences, return_tensors="pt", padding=True)
    mask = tokens.attention_mask.unsqueeze(-1).repeat(1, 1, sae.cfg.d_sae).to(device)
    n_tokens = mask.sum().item()

    _, cache = lm.run_with_cache(sentences)
    inputs = cache[sae.cfg.hook_name]
    latents = sae.encode(inputs)
    recons = sae.decode(latents)

    if args.model == "gpt2":
        latent_acts = latents[:, 1:] * mask
        recon_mse = ((inputs - recons)**2).sum(dim=-1)[:, 1:] * mask[:, :, 0]
    elif args.model == "pythia":
        latent_acts = latents * mask
        recon_mse = ((inputs - recons)**2).sum(dim=-1) * mask[:, :, 0]
    else:
        raise ValueError(f"{args.model = }")
    
    return latent_acts, recon_mse, n_tokens

df = pl.DataFrame(
    schema=dict(
        model=pl.String,
        lang=pl.String,
        layer=pl.String,
        latent_acts=pl.Array(pl.Float64, d_sae),
        latent_count=pl.Array(pl.Int64, d_sae),
        recon_mse=pl.Float64,
        n_tokens=pl.Int64,
    )
)

for sae_id in sae_ids:
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=sae_release, 
        sae_id=sae_id,
        device=device,
    )

    for lang, sentences in tqdm(lang_sentences.items(), desc=sae_id):
        latent_acts = t.zeros(sae.cfg.d_sae, dtype=t.float64).to(device)
        latent_count = t.zeros(sae.cfg.d_sae, dtype=t.int64).to(device)
        recon_mse = 0
        n_tokens = 0
    
        dataloader = DataLoader(sentences, batch_size=10)
        
        for batch in dataloader:
            batch_latents, batch_recon_mse, batch_n_tokens = run_sae(sae, batch)
            
            latent_acts += batch_latents.sum(dim=[0, 1])
            latent_count += (batch_latents > 0).sum(dim=[0, 1])
            recon_mse += batch_recon_mse.sum()
            n_tokens += batch_n_tokens
    
        df.extend(pl.DataFrame(dict(
            model=[args.model],
            lang=[lang],
            layer=[sae_id],
            latent_acts=[latent_acts.cpu().numpy()],
            latent_count=[latent_count.cpu().numpy()],
            recon_mse=[recon_mse],
            n_tokens=[n_tokens],
        )))

    del sae, cfg_dict, sparsity

df.write_json(f"results/metrics-{args.model}.json")
