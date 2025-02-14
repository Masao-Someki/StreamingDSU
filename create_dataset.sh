#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpuA100x4
#SBATCH --account=bbjs-delta-gpu --reservation sup-10124
#SBATCH -t 48:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

config=
expdir=

. path.sh
. parse_options.sh


for split in train; do
    python create_dataset.py \
        --config ${config} \
        --ckpt ${expdir}/latest.pth \
        --dump_dir ${expdir}/ \
        --dataset_split ${split}
done

