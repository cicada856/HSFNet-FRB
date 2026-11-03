#!/usr/bin/env bash
set -eo pipefail
export PATH=/home/linzhizhen/miniconda3/envs/frabjous/bin:$PATH
export LD_LIBRARY_PATH=/home/linzhizhen/miniconda3/envs/frabjous/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1
cd /home/linzhizhen/HSFNet-FRB
exec /home/linzhizhen/miniconda3/envs/frabjous/bin/python -u scripts/run_matched_simulation_only_dlsr_control.py --report-dir reports/matched_simulation_only_dlsr_control --artifact-dir artifacts/runs/matched_simulation_only_dlsr_control --device gpu --gpu-id 1
