#!/bin/bash
#SBATCH --output=logs/%j-%x.log
#SBATCH --error=logs/%j-%x.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100x4,gpuA40x4
#SBATCH --account=bbjs-delta-gpu --reservation sup-10124

# . path.sh
set -x
layer=
frame=

. parse_options.sh

echo $layer
echo $frame

echo "START TRAIN $(date '+%Y-%m-%d %H:%M:%S')"
config=config/final_orig_bpe/unit2text_final.yaml
LAYER=$layer FRAME=$frame HF_HOME=/work/hdd/bbjs/kchoi1/StreamingDSU/download/hf \
    envs/bin/python3 train.py \
        --train_u2t \
        --train_config $config \
        --stats_dir u2t_stats/stats_$layer\_$frame/

echo "START EVALUATE $(date '+%Y-%m-%d %H:%M:%S')"
model_dir=$(find exp/unit2text_final_$layer\_$frame/ -type f -name "checkpoint.pth" | head -n 1)
model_dir=$(dirname "$model_dir")

for split in test_clean test_other test_1h; do
    LAYER=$layer FRAME=$frame HF_HOME=/work/hdd/bbjs/kchoi1/StreamingDSU/download/hf \
        envs/bin/python3 evaluate_unit2text.py \
        --config $config \
        --output_dir $model_dir \
        --unit_dir download/dump/l$layer\_$frame\_$frame \
        --split $split \
        --mt_config $model_dir/config.yaml \
        --mt_model $model_dir/valid.acc.best.pth
done
