import json
from tqdm import tqdm

from src.data.parse_mcq import parse_mcq_output


INPUT_PATH = "data/mcq_raw/prelim_generations.jsonl"
OUTPUT_PATH = "data/processed/mcq_parsed.jsonl"


def run_parsing():
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    parsed = []
    failures = 0

    for item in tqdm(data, desc="Parsing MCQs"):
        result = parse_mcq_output(item["raw_output"])

        if result["status"] == "success":
            parsed.append({
                "example_id": item["example_id"],
                "question": item["question"],
                "answer": item["answer"],
                "options": result["options"],
                "correct": result["correct"]
            })
        else:
            failures += 1

    # save
    with open(OUTPUT_PATH, "w") as f:
        for rec in parsed:
            f.write(json.dumps(rec) + "\n")

    print("\nParsed:", len(parsed))
    print("Failures:", failures)


if __name__ == "__main__":
    run_parsing()