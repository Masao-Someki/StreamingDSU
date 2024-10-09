#!/bin/bash

. tools/activate_python.sh


python train.py \
    --task "asr" \
    --model_config "conf_model/wavlm_baseline.yaml" \
    --train_config "conf_train/sample.yaml" \
    --ngpu 1 \
    --evaluate \
    --exp_dir "exp/wavlm_baseline" 