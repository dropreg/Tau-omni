import re


JUDGE_USER_PROMPT = """请比较下面同一用户问题的两个候选回答，并判断哪个回答整体更好。

[User Query]
{query}

[Assistant A]
{response_1}

[Assistant B]
{response_2}

请先给出简要分析，最后单独输出一行：
Winner: [[A]] / [[B]] / [[Tie]]"""


VERDICT_PATTERN = re.compile(r"\[\[\s*(A|B|Tie)\s*\]\]", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    matches = VERDICT_PATTERN.findall(text)
    if not matches:
        return None

    verdict = matches[-1].strip().lower()
    if verdict == "tie":
        return "Tie"
    if verdict == "a":
        return "A"
    if verdict == "b":
        return "B"
    return None
