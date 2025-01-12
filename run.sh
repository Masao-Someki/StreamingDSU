#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

layer=
train_config=config/soundstream/soundstream.yaml

. path.sh
. parse_options.sh

python train.py --train_config ${train_config}
