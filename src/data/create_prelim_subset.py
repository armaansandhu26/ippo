import json
import random


INPUT_PATH = "data/raw/gsm8k_base.jsonl"
OUTPUT_PATH = "data/raw/gsm8k_prelim_subset.jsonl"

SUBSET_SIZE = 1500
SEED = 42


def create_subset():
    # load dataset
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    # deterministic shuffle
    random.seed(SEED)
    random.shuffle(data)

    # select subset
    subset = data[:SUBSET_SIZE]

    # save
    with open(OUTPUT_PATH, "w") as f:
        for rec in subset:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(subset)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    create_subset()