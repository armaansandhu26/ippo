import argparse
import json
import random
from tqdm import tqdm


DEFAULT_INPUT_PATH = "data/processed/mcq_judge_passed.jsonl"
TRAIN_OUTPUT = "data/processed/prelim_train.jsonl"
TEST_OUTPUT = "data/processed/prelim_test.jsonl"

SEED = 42


def force_correct_A(options, correct):
    correct_value = options[correct]

    distractors = [v for k, v in options.items() if k != correct]

    new_options = {
        "A": correct_value,
        "B": distractors[0],
        "C": distractors[1],
        "D": distractors[2],
    }

    return new_options, "A"


def shuffle_options(options, correct):
    items = list(options.items())
    random.shuffle(items)

    new_options = {}
    new_correct = None

    for idx, (k, v) in enumerate(items):
        new_key = ["A", "B", "C", "D"][idx]
        new_options[new_key] = v

        if k == correct:
            new_correct = new_key

    return new_options, new_correct


def run_transformation(input_path: str):
    random.seed(SEED)

    with open(input_path) as f:
        data = [json.loads(line) for line in f]

    train_data = []
    test_data = []

    for item in tqdm(data, desc="Transforming"):

        options = item["options"]
        correct = item["correct"]
        split = item["example_id"].split("_")[1]  # train/test

        if split == "train":
            new_options, new_correct = force_correct_A(options, correct)

            train_data.append({
                **item,
                "options": new_options,
                "correct": new_correct
            })

        else:
            new_options, new_correct = shuffle_options(options, correct)

            test_data.append({
                **item,
                "options": new_options,
                "correct": new_correct
            })

    # save
    with open(TRAIN_OUTPUT, "w") as f:
        for rec in train_data:
            f.write(json.dumps(rec) + "\n")

    with open(TEST_OUTPUT, "w") as f:
        for rec in test_data:
            f.write(json.dumps(rec) + "\n")

    print("\nTrain size:", len(train_data))
    print("Test size:", len(test_data))
    print("Input:", input_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split into train (A=correct) / test (shuffled)")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Validated MCQs (default: LLM judge-passed file {DEFAULT_INPUT_PATH})",
    )
    args = parser.parse_args()
    run_transformation(args.input)
