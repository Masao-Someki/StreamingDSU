#!/bin/bash

# for i in 6 9 12 15 18 21; do
# for i in 21; do
#     # sbatch --job-name=ft_u2t_newbpe-$i exp_layerwise_newbpe_ft.sh
#     sbatch --job-name=ft_u2t_origbpe-$i exp_layerwise_origbpe_ft.sh
# done

# for layer in 12 15 18 21; do
for layer in 21; do
    for frame in 0 1 2 4 8 16 32 64; do
        sbatch --job-name=u2t-$layer-$frame \
            exp_final_U2T.sh \
                --layer $layer \
                --frame $frame
    done
done
