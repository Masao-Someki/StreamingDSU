#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH -t 10:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# python evaluate.py --config config/streaming_wavlm/future_20.yaml --ckpt exp/future_20/20250113_165742/valid.acc.best.pth --output_dir future_20
# python evaluate.py --config config/streaming_wavlm/future_10.yaml --ckpt exp/future_10/20250113_165815/valid.acc.best.pth --output_dir future_10
# python evaluate.py --config config/streaming_wavlm/future_5.yaml --ckpt exp/future_5/20250113_165815/valid.acc.best.pth --output_dir future_5
# python evaluate.py --config config/streaming_wavlm/streaming_wavlm.yaml --ckpt exp/streaming_wavlm/20250113_165414/valid.acc.best.pth --output_dir streaming_wavlm

python evaluate.py --config config/streaming_wavlm/future_1.yaml --ckpt exp/future_1/20250114_100406/valid.acc.best.pth --output_dir future_1
python evaluate.py --config config/streaming_wavlm/future_3.yaml --ckpt exp/future_3/20250114_191518/valid.acc.best.pth --output_dir future_3
python evaluate.py --config config/streaming_wavlm/future_5.yaml --ckpt exp/future_5/20250113_165815/valid.acc.best.pth --output_dir future_5
python evaluate.py --config config/streaming_wavlm/future_5_past_5.yaml --ckpt exp/future_5_past_5/20250114_194913/valid.acc.best.pth --output_dir future_5_past_5
python evaluate.py --config config/streaming_wavlm/future_5_past_10.yaml --ckpt exp/future_5_past_10/20250114_200900/valid.acc.best.pth --output_dir future_5_past_10
python evaluate.py --config config/streaming_wavlm/future_5_past_20.yaml --ckpt exp/future_5_past_20/20250114_203204/valid.acc.best.pth --output_dir future_5_past_20
python evaluate.py --config config/streaming_wavlm/future_5_past_40.yaml --ckpt exp/future_5_past_40/20250115_094925/valid.acc.best.pth --output_dir future_5_past_40
