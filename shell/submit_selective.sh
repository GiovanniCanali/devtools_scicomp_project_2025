#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python scripts/run.py --config_file experiments/selective_copy/s6.yaml --fit True --test True --test True
python scripts/run.py --config_file experiments/selective_copy/mamba_s6.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/s4d_inv.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/s4d_real.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/mamba_s4_low_rank.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/mamba_s4d_real.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/mamba_s4d_inv.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/s4_low_rank.yaml --fit True --test True
python scripts/run.py --config_file experiments/selective_copy/s4.yaml --fit True --test True