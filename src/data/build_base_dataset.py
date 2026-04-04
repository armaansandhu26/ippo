import json
from datasets import load_dataset
from tqdm import tqdm


def extract_final_answer(raw_answer):
    """
    Extract final answer after #### with minimal processing.
    """

    # Step 1: extract after ####
    if "####" in raw_answer:
        answer = raw_answer.split("####")[-1].strip()
    else:
        answer = raw_answer.strip()

    # Step 2: light normalization (ONLY this)
    answer_clean = answer.replace(",", "").strip()

    # Step 3: numeric conversion (safe attempt)
    try:
        numeric = float(answer_clean)
        status = "success"
    except:
        numeric = None
        status = "text_only"

    return answer_clean, numeric, status, answer


def build_base_dataset(output_path="data/raw/gsm8k_base.jsonl"):
    dataset = load_dataset("gsm8k", "main")

    with open(output_path, "w") as f:

        for split in ["train", "test"]:
            data = dataset[split]

            for idx, example in enumerate(tqdm(data, desc=f"{split}")):
                question = example["question"]
                raw_answer = example["answer"]

                final_text, final_numeric, status, raw_extracted = extract_final_answer(raw_answer)

                record = {
                    "example_id": f"gsm8k_{split}_{idx:06d}",
                    "split": split,
                    "question": question,
                    "raw_answer": raw_answer,

                    # core fields
                    "final_answer_text": final_text,
                    "final_answer_numeric": final_numeric,
                    "answer_extraction_status": status,

                    # debug field (KEEP THIS)
                    "answer_raw_extracted": raw_extracted
                }

                f.write(json.dumps(record) + "\n")

    print("\nSaved base dataset to:", output_path)


if __name__ == "__main__":
    build_base_dataset()