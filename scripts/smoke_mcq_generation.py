"""Print a few GSM8K MCQ generations using build_mcq_prompt (dev smoke test; not the judge)."""

from datasets import load_dataset
from dotenv import load_dotenv
import os
from openai import OpenAI

from src.prompts.mcq_prompt import build_mcq_prompt

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def main():
    dataset = load_dataset("gsm8k", "main")
    sample = dataset["train"].select(range(5))

    for i, ex in enumerate(sample):
        question = ex["question"]
        answer = ex["answer"].split("####")[-1].strip()
        prompt = build_mcq_prompt(question, answer)

        print("\n" + "=" * 60)
        print(f"Example {i}")
        print("=" * 60)
        print("\nQUESTION:")
        print(question)
        print("\nTRUE ANSWER:")
        print(answer)
        print("\nMODEL OUTPUT:")

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
