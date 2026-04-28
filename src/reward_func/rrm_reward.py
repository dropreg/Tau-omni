import re


def extract_answer(response: str, answer: int):

    assert type(answer) == int, f"The final answer should be an integer, not [{type(answer)}]"
    
    pred = None

    pattern = r'\[{1,2}([abcABC])\]{1,2}'
    matches = re.findall(pattern, response)
    
    if len(matches) > 0:
        pred = matches[-1].upper()
    
    correct = False
    mapping = {0: 'A', 1: 'B', 2: 'Tie'}
    if pred == mapping.get(answer):
        correct = True
    
    return correct, pred

def accuracy_reward(solution_str: str, answer: int) -> float:

    correct, _ = extract_answer(response=solution_str, answer=answer)
    return 1.0 if correct else 0.0

def format_reward(solution_str: str) -> float:

    pattern = r'\[{1,2}([abcABC])\]{1,2}'
    if re.search(pattern, solution_str):
        return 1.0
    return 0.0

def rrm_reward_function(
    data_source: str,
    solution_str: str,
    ground_truth: int,
    extra_info: dict,
    **kwargs
):

    acc_reward = accuracy_reward(solution_str=solution_str, answer=ground_truth)
    
    frm_reward = format_reward(solution_str=solution_str)
    
    if frm_reward:
        return acc_reward * 0.5 + 0.5
    else:
        return 0.0
