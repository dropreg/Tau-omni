import re


def extract_omni_answer(response: str, answer: int):
    
    assert type(answer) == int, f"The final answer shoule be a interger. not [{type(answer)}]"
    assert answer in [0, 1], f"The final answer shoule be 0 or 1, not [{answer}]"

    pred = None
    pattern = r'\[{1,2}([abcABC])\]{1,2}'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        pred = matches[-1]
    else:
        pred = None
    if isinstance(pred, str):
        pred = pred.upper()
    
    correct = False
    if pred == 'A' and answer == 0:
        correct = True
    elif pred == 'B' and answer == 1:
        correct = True
    elif pred == 'C' and answer == 2:
        correct = True
    
    return correct, pred


def accuracy_reward(solution_str: str, answer: int) ->float:

    correct, pred = extract_omni_answer(response=solution_str, answer=answer)
    
    if correct:
        return 1.0
    else:
        return 0.0

def rule_reward(
    data_source: str,
    solution_str: str,
    ground_truth: int,
    extra_info: dict,
    **kwargs
):

    reward = 0
    reward += accuracy_reward(solution_str=solution_str, answer= ground_truth)
    return reward
