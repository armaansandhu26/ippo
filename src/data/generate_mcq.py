import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI

from src.prompts.mcq_prompt import build_mcq_prompt


# load env
load_dotenv()

# Per-request ceiling so a single hung API call does not stall the whole run (override via env).
_GENERATE_TIMEOUT = float(os.getenv("OPENAI_GENERATE_TIMEOUT", "120"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=_GENERATE_TIMEOUT)


INPUT_PATH = "data/raw/gsm8k_prelim_subset.jsonl"
OUTPUT_PATH = "data/mcq_raw/prelim_generations.jsonl"


def generate_mcq():
    # load subset
    with open(INPUT_PATH) as f:
        data = [json.loads(line) for line in f]

    os.makedirs("data/mcq_raw", exist_ok=True)

    with open(OUTPUT_PATH, "w") as out_f:

        for item in tqdm(data, desc="Generating MCQs"):
            question = item["question"]
            answer = item["final_answer_text"]

            prompt = build_mcq_prompt(question, answer)

            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )

                raw_output = response.choices[0].message.content

            except Exception as e:
                raw_output = f"ERROR: {str(e)}"

            record = {
                "example_id": item["example_id"],
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "raw_output": raw_output
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()


if __name__ == "__main__":
    generate_mcq()