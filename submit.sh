#!/bin/bash
#SBATCH -N 1
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

. path.sh

python train.py --train_config config/asr_convrnn.yaml
