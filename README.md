# StreamingDSU
Course project for on-device project

## Installation (conda)
```sh
conda create -p ./envs python=3.10
conda activate ./envs
conda install -y pytorch=2.4.0 torchaudio=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install git+https://github.com/espnet/espnet.git
pip install git+https://github.com/s3prl/s3prl.git
pip install -r tools/requirements.txt
```
