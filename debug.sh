#!/bin/bash

. tools/activate_python.sh


python train.py \
    --task "asr" \
    --model_config "conf_model/sample_dh.yaml" \
    --train_config "conf_train/sample.yaml" \
    --ngpu 1 \
    --run_train \
    --evaluate \
    --eval_quantize \
    --export_onnx \
    --ckpt "exp/asr_streaming_dsu/5epoch.pth" \
    --quantize_config "conf_quantize/sample.yaml" \
    --debug