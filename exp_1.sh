#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 24:00:00
# SBATCH --gpus=v100-16:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=7G

. path.sh

layer=20
train_conf_name=
task=
model=wavlm

. tools/parse_options.sh

mkdir -p exp_1/${train_conf_name}_layer${layer}_${model}
python train.py \
    --task ${task} \
    --model_config "conf_model/exp_1/${model}_${layer}.yaml" \
    --train_config "conf_train/exp_1/${train_conf_name}.yaml" \
    --ngpu 1 \
    --evaluate \
    --exp_dir "exp_1/${train_conf_name}_layer${layer}_${model}"  > exp_1/${train_conf_name}_layer${layer}_${model}/train.log

    # --run_train \