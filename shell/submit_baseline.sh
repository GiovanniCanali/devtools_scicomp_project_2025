#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python scripts/run.py --config_file experiments/s4d_inv.yaml --fit True --test True
python scripts/run.py --config_file experiments/s4d_real.yaml --fit True --test True
python scripts/run.py --config_file experiments/s6.yaml --fit True --test True --test True
python scripts/run.py --config_file experiments/s4_low_rank.yaml --fit True --test True
python scripts/run.py --config_file experiments/s4.yaml --fit True --test True