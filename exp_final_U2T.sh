#!/bin/bash
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:1
# SBATCH --partition=gpuA40x4
# SBATCH --account=bbjs-delta-gpu
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --account=bbjs-delta-gpu --reservation sup-10124

. path.sh
set -x
layer=
frame=

. parse_options.sh

export LAYER=$layer
export FRAME=$frame

echo "START TRAIN $(date '+%Y-%m-%d %H:%M:%S')"
config=config/final_orig_bpe/unit2text_final.yaml
HF_HUB_CACHE=/u/someki1/workspace/hub \
    python train.py \
       --train_config $config \
       --skip_collect_stats

echo "START EVALUATE $(date '+%Y-%m-%d %H:%M:%S')"
model_dir=$(find exp/unit2text_final_$layer_$frame/ -type f -name "checkpoint.pth" | head -n 1)
model_dir=$(dirname "$model_dir")

audio2unit_dir=$(find exp/wavlm_weighted_trainable_$layer/ -type f -name "checkpoint.pth" | head -n 1)
audio2unit_dir=$(dirname "$model_dir")

for split in test_clean test_other test_1h; do
    HF_HUB_CACHE=/u/someki1/workspace/hub \
        python evaluate_unit2text.py \
        --config $config \
        --output_dir $model_dir \
        --dataset_split $split \
        --mt_config $model_dir/config.yaml \
        --mt_model $model_dir/valid.acc.best.pth
done