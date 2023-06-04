#!/bin/sh

nohup python -u main_sac.py --env_id "Hopper-v4" --log_alpha_lr 0.001 --log_path "Hopper-v4-alphatuning/" > Hopper-v4-alphatuning.out &
wait
# nohup python -u main_sac.py --env_id "Hopper-v4" --log_path "Hopper-v4-noalphatuning/" > Hopper-v4-noalphatuning.out &
# wait
nohup python -u main_sac.py --env_id "HalfCheetah-v4" --log_alpha_lr 0.001 --log_path "HalfCheetah-v4-alphatuning/" > HalfCheetah-v4-alphatuning.out &
# wait
# nohup python -u main_sac.py --env_id "HalfCheetah-v4" --log_path "HalfCheetah-v4-noalphatuning/" > HalfCheetah-v4-noalphatuning.out &