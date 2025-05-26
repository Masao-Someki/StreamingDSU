# On-device Streaming Discrete Speech Units
- Accepted to Interspeech 2025
- Additional results (Baseline, SSL Frozen, SSL FT in Figure 4) can be found here: https://github.com/juice500ml/espnet/tree/dsu_baseline/egs2/interspeech2024_dsu_challenge

## Installation (conda)
```sh
conda create -p ./envs python=3.10
conda activate ./envs
conda install -y pytorch=2.4.0 torchaudio=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```
