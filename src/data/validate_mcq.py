import json
from tqdm import tqdm


INPUT_PATH = "data/processed/mcq_parsed.jsonl"
OUTPUT_PATH = "data/processed/mcq_validated.jsonl"


def is_numeric(x):
    try:
        float(x)
        return True
    except:
        return False


def validate_record(record):
    options = record["options"]
    correct = record["correct"]
    answer = record["answer"]

    # check 1: 4 options
    if len(options) != 4:
        return False, "not_4_options"

    # check 2: all numeric
    if not all(is_numeric(v) for v in options.values()):
        return False, "non_numeric_option"

    # check 3: uniqueness
    if len(set(options.values())) != 4:
        return False, "duplicate_options"

    # check 4: correct answer matches
    try:
        if float(options[correct]) != float(answer):
            return False, "incorrect_label"
    except:
        return False, "conversion_error"

    return True, "valid"


def run_validation():
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    valid = []
    stats = {}

    for item in tqdm(data, desc="Validating MCQs"):
        is_valid, reason = validate_record(item)

        stats[reason] = stats.get(reason, 0) + 1

        if is_valid:
            valid.append(item)

    # save valid data
    with open(OUTPUT_PATH, "w") as f:
        for rec in valid:
            f.write(json.dumps(rec) + "\n")

    print("\nValidation Results:")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print(f"\nValid samples: {len(valid)} / {len(data)}")


if __name__ == "__main__":
    run_validation()