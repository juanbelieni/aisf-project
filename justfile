minify-metrics:
    #!/usr/bin/env -S parallel --shebang --ungroup --jobs {{ num_cpus() }}
    jq --arg digits 2 '($digits | tonumber) as $d | walk(if type == "number" then (. * pow(10;$d) | ceil) / pow(10;$d) else . end)' -c results/metrics-gpt2.json > results/metrics-gpt2.min.json
    jq --arg digits 2 '($digits | tonumber) as $d | walk(if type == "number" then (. * pow(10;$d) | ceil) / pow(10;$d) else . end)' -c results/metrics-pythia.json > results/metrics-pythia.min.json
