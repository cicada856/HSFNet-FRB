#!/usr/bin/env bash
set -euo pipefail

export PATH=/home/linzhizhen/miniconda3/envs/frabjous/bin:${PATH}
export LD_LIBRARY_PATH=/home/linzhizhen/miniconda3/envs/frabjous/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_VISIBLE_DEVICES=1
cd /home/linzhizhen/HSFNet-FRB
exec /home/linzhizhen/HSFNet-FRB/reports/decoupled_cost_sensitive_hard_router/reproduce.sh
