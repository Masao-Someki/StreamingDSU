#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpuA40x4
#SBATCH --account=bbjs-delta-gpu
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


. path.sh
. parse_options.sh

python train_kmeans.py \
    --savedir km_model \
    --layer 21 \
    --n_clusters 2000

python train_kmeans.py \
    --savedir km_model \
    --layer 21 \
    --n_clusters 1000

python train_kmeans.py \
    --savedir km_model \
    --layer 21 \
    --n_clusters 500

python train_kmeans.py \
    --savedir km_model \
    --layer 6 \
    --n_clusters 2000

python train_kmeans.py \
    --savedir km_model \
    --layer 12 \
    --n_clusters 2000

python train_kmeans.py \
    --savedir km_model \
    --layer 18 \
    --n_clusters 2000
