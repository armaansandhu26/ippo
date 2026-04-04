"""User message for the MCQ *generation* step (build 4 options from Q + answer)."""

def build_mcq_prompt(question, answer):
    return f"""
You are given a math word problem and its correct final answer.

Your task is to convert it into a multiple-choice question.

INSTRUCTIONS:

- First, internally think about the key steps needed to solve the problem.
- Then think of 3 specific mistakes someone could make in those steps.

The incorrect answers MUST:
- Correspond to those specific mistakes
- Be values someone would actually compute if they made that mistake
- NOT be arbitrary numbers or simple multiples

STRICT REQUIREMENTS:
- Provide exactly 4 options labeled A, B, C, D
- Exactly ONE option must be correct
- Do NOT use evenly spaced numbers or simple patterns (like 5, 10, 15, 20)
- Do NOT include explanations or reasoning in the output
- All options should be similar in scale and format

Question:
{question}

Correct Answer:
{answer}

Output format (STRICT):

A. <option>
B. <option>
C. <option>
D. <option>
Correct Answer: <A/B/C/D>
"""