# On-device Streaming Discrete Speech Units

- **Accepted to Interspeech 2025**
- Additional results (Baseline, SSL Frozen, SSL FT in Figure 4) can be found here: [espnet/dsu_baseline](https://github.com/juice500ml/espnet/tree/dsu_baseline/egs2/interspeech2024_dsu_challenge)

## Overview

This repository provides code and scripts for streaming discrete speech unit (DSU) extraction and evaluation, targeting on-device scenarios. The workflow is organized into several stages, each with corresponding shell scripts for easy execution. This document details each stage and how to run it.

---

## Installation

We recommend using Conda for environment management.

```sh
conda create -p ./envs python=3.10
conda activate ./envs
conda install -y pytorch=2.4.0 torchaudio=2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Project Structure

- **Shell scripts** are provided to automate most stages.  
- Core Python scripts implement dataset creation, training, evaluation, and analysis.

Key files and directories include:
- `create_dataset.sh` — Dataset creation
- `train_kmeans.sh` — K-means clustering for unit extraction
- `run.sh` — Main training
- `evaluate.sh` — Model evaluation
- `submit.sh`, `submit_everything.sh` — Submission utilities
- `config/` — YAML configuration files
- `src/`, `egs/`, `unit_analyze/`, etc. — Code and experiments

---

## Stage-by-Stage Usage

### 1. Dataset Creation

Prepare the required dataset by running:

```sh
bash create_dataset.sh
```
- This will invoke `create_dataset.py` and process raw audio/text into the required format.
- Adjust the script or configs as needed for your dataset location.

### 2. K-means Training (Unit Extraction)

To perform K-means clustering on features (for discrete unit extraction):

```sh
bash train_kmeans.sh
```
- This will run `train_kmeans.py` using parameters set in the shell script.

### 3. Main Model Training

Train your streaming DSU model by running:

```sh
bash run.sh
```
- This runs `train.py` using the configuration specified in `run.sh` (default: `config/soundstream/soundstream.yaml`).
- You can override parameters by editing the script or passing options as environment variables.

### 4. Model Evaluation

Evaluate the trained model:

```sh
bash evaluate.sh
```
- This will call `evaluate.py` and/or `evaluate_unit2text.py` to compute metrics and generate results.

### 5. Submission

For challenge or benchmark submissions:

```sh
bash submit.sh
# or
bash submit_everything.sh
```
- These scripts package results for submission or evaluation.

---

## Configuration

- All major stages reference configuration files under `config/`.
- Edit YAML files (e.g., `soundstream.yaml`) to change model, training, or preprocessing parameters.

---

## Useful Scripts

- `calc_gflops.py` — Measure model computational cost.
- `visualize_cooccurrence.py` — Analyze and visualize unit co-occurrences.
- `grid2tsv.py` — Convert alignment grids for analysis.

---

## Tips

- Each shell script sets up the environment via `path.sh` and parses options with `parse_options.sh`.
- GPU resources are managed via SLURM directives in the scripts; adjust as needed for your compute cluster.

---

## Additional Notes

- For further details on baseline and ablation results, see: [espnet/dsu_baseline](https://github.com/juice500ml/espnet/tree/dsu_baseline/egs2/interspeech2024_dsu_challenge).
- For the complete list of files and scripts, visit the [repository contents](https://github.com/Masao-Someki/StreamingDSU/tree/master).

---

## Citation

If you use this repository, please cite our Interspeech 2025 paper.

```
@inproceedings{choi25_interspeech,
  title={On-device Streaming Discrete Speech Units},
  author={Kwanghee Choi and Masao Someki and Emma Strubell and Shinji Watanabe},
  booktitle={Interspeech},
  year={2025}
}
```

---

For questions, please open an issue or contact the authors.
