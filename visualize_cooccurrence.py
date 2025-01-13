#!/usr/bin/env python3

# This script visualizes the co-occurrence between phones and units.
# Copyright 2024 Shuichiro Shimizu
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser


def joint_occurrence_to_conditional_prob(C: np.ndarray) -> np.ndarray:
    """
    Convert joint occurrence matrix to conditional probability matrix
    """
    sum_C = np.sum(C, axis=0)[:, np.newaxis]
    p_phone_given_unit = np.divide(C.T, sum_C, where=sum_C != 0).T

    return p_phone_given_unit


def get_cooccurrence_matrix(
    phones: list[str], units: list[int]
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Get co-occurrence matrix between phones and units
    """
    cooccurrences = Counter(zip(phones, units))

    phone_classes = list(set(phones))
    unit_classes = list(set(units))

    M = len(phone_classes)
    N = len(unit_classes)

    C = np.zeros((M, N), dtype=int)

    for (phone, unit), count in cooccurrences.items():
        i = phone_classes.index(phone)
        j = unit_classes.index(unit)
        C[i][j] = count

    return np.array(C), phone_classes, unit_classes


def read_tsv(tsv_file: str) -> list[str]:
    """
    Read tsv file and return a list of sequences
    """
    with open(tsv_file) as f:
        lines = [l.strip() for l in f.readlines()]

    seqs = []
    for line in lines:
        _utt_id, seq = line.split("\t")
        seq = seq.split(" ")
        seqs.extend(seq)

    return seqs


def sort_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Sort rows of a matrix based on the sum of each row
    """
    # Get the sum of each row
    sum_for_each_row = np.sum(matrix, axis=1)

    # Sort rows based on the sum
    sorted_row_idx = np.argsort(-sum_for_each_row, kind="stable")

    return sorted_row_idx


def get_sorted_groups(matrix: np.ndarray) -> dict[np.ndarray, list[int]]:
    """
    Sort columns of a matrix based on the maximum value in each column
    """
    # Get the row indices of the maximum values for each column
    max_row_indices = np.argmax(matrix, axis=0)

    # Group columns based on max row indices
    grouped_columns = defaultdict(list)
    for col_idx, row_idx in enumerate(max_row_indices):
        grouped_columns[row_idx].append(col_idx)

    # Sort columns within each group by their values
    sorted_grouped_columns = {}
    for row_idx, col_indices in grouped_columns.items():
        sorted_col_indices = sorted(
            col_indices, key=lambda col_idx: matrix[row_idx, col_idx], reverse=True
        )
        sorted_grouped_columns[row_idx] = sorted_col_indices

    return sorted_grouped_columns


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--phones-tsv",
        type=str,
        required=True,
        help="tsv file containing phone sequences. First column is utterance id, second column is phone sequence separated by spaces.",
    )
    parser.add_argument(
        "--units-tsv",
        type=str,
        required=True,
        help="tsv file containing unit sequences. First column is utterance id, second column is unit sequence separated by spaces.",
    )
    parser.add_argument(
        "--figure-dir", type=str, required=True, help="directory to save the figure"
    )
    args = parser.parse_args()

    phones_tsv = args.phones_tsv
    units_tsv = args.units_tsv

    exp_id = Path(units_tsv).stem

    phones = read_tsv(phones_tsv)
    units = read_tsv(units_tsv)
    units = list(map(int, units))

    C, phone_classes, unit_classes = get_cooccurrence_matrix(phones, units)
    C = joint_occurrence_to_conditional_prob(C)

    sorted_row_idx = sort_rows(C)
    sorted_phones = [phone_classes[i] for i in sorted_row_idx]

    sorted_col_idx = []
    sorted_groups = get_sorted_groups(C)
    for i in sorted_row_idx:
        sorted_col_idx.extend(sorted_groups.get(i, []))

    D = np.zeros_like(C)
    for new_i, i in enumerate(sorted_row_idx):
        for new_j, j in enumerate(sorted_col_idx):
            D[new_i, new_j] = C[i, j]

    fig, ax = plt.subplots(figsize=(18, 12))
    cax = ax.imshow(D, cmap="Blues", aspect="auto", interpolation="nearest")
    fig.colorbar(cax)

    ax.set_yticks(np.arange(len(sorted_phones)))
    ax.set_yticklabels(sorted_phones)

    ax.set_title(
        f"{exp_id} ({len(phone_classes)} active phones, {len(unit_classes)} active units)"
    )

    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_dir / f"{exp_id}.png")


if __name__ == "__main__":
    main()