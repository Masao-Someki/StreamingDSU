import argparse
import json
import datasets
import editdistance
from itertools import groupby
import numpy as np

import string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model.")
    parser.add_argument("--exp_dir", required=True, help="Path to the exp directory.")
    args = parser.parse_args()

    dataset = datasets.load_from_disk("asr_data")
    valid_ds = dataset["test_clean"]

    with open(f"{args.exp_dir}/results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    cers = []
    cer_unit = []
    cer_dedup = []
    for i in range(len(results)):
        hyp = results[i]["hyp"].lower().translate(str.maketrans('', '', string.punctuation))
        # ref = valid_ds[i]["text"].lower().translate(str.maketrans('', '', string.punctuation))
        ref = results[i]["ref"].lower().translate(str.maketrans('', '', string.punctuation))
        cer = editdistance.eval(hyp, ref) / len(ref)
        cers.append(cer)

        cer_unit.append(
            editdistance.eval(
                results[i]["hyp_units"],
                results[i]["ref_units"]
            ) / len(results[i]["ref_units"])
        )
        acc = []
        for j in range(len(results[i]["hyp_units"])):
            if results[i]["hyp_units"][j] == results[i]["ref_units"][j]:
                acc.append(1)
            else:
                acc.append(0)
        

        cer_dedup.append(
            editdistance.eval(
                results[i]["hyp_units_dedup"],
                ''.join([x[0] for x in groupby(results[i]["ref_units"])])
            ) / len(results[i]["ref_units"])
        )

    print(f"CER: {np.mean(cers):.2%}")
    print(f"CER (units): {np.mean(cer_unit):.2%}")
    print(f"ACC (units): {np.mean(acc):.2%}")
    print(f"CER (deduplicated units): {np.mean(cer_dedup):.2%}")
