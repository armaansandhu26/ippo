import re


def parse_mcq_output(raw_output):
    """
    Parses LLM MCQ output into structured format.

    Returns:
        {
            "options": {...},
            "correct": "A",
            "status": "success" / "fail"
        }
    """

    try:
        lines = raw_output.strip().split("\n")

        options = {}
        correct = None

        # normalize lines
        lines = [line.strip() for line in lines if line.strip()]

        # parse options
        for line in lines:
            if re.match(r"^[A-D]\.", line):
                key = line[0]
                value = line[2:].strip()
                options[key] = value

            elif "Correct Answer" in line:
                correct = line.split(":")[-1].strip()

        # validation
        if len(options) != 4:
            return {"status": "fail", "reason": "not_4_options"}

        if correct not in ["A", "B", "C", "D"]:
            return {"status": "fail", "reason": "invalid_correct"}

        return {
            "options": options,
            "correct": correct,
            "status": "success"
        }

    except Exception as e:
        return {"status": "fail", "reason": str(e)}