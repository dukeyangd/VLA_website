#!/usr/bin/env bash
# Deprecated alias: pick-tissue-named entry → generic sonic latent sim eval.
# Prefer: tools/eval/run_sonic_latent_sim_eval.sh
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sonic_latent_sim_eval.sh" "$@"
