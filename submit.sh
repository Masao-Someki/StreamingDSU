#!/bin/bash

for l in $(seq 6); do
    sbatch exp_1.sh \
        --train_conf_name lr0001_step09 \
        --layer ${l} \
        --task "tts" \
        --model "hubert"
done
