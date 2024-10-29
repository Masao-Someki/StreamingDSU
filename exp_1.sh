#!/bin/bash
# SBATCH -N 1
# SBATCH -p GPU-shared
# SBATCH -t 24:00:00
# SBATCH --gpus=v100-16:1
# SBATCH --ntasks-per-node=16

. path.sh

layer=20

. tools/parse_options.sh

python train.py \
    --task "asr" \
    --model_config "conf_model/exp_1/wavlm_${layer}.yaml" \
    --train_config "conf_train/exp_1.yaml" \
    --ngpu 1 \
    --run_train \
    --exp_dir "exp_1/wavlm_${layer}" 
