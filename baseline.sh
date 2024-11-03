#!/bin/bash

. tools/activate_python.sh


python train.py \
    --task "asr" \
    --model_config "conf_model/exp_1/wavlm_1.yaml" \
    --train_config "conf_train/sample.yaml" \
    --ngpu 0 \
    --evaluate \
    --exp_dir "exp/lr0001_step09_layer1_wavlm" 