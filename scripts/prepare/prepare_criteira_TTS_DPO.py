import pdb
import json
import re
import random


rubric_prompt = """Given a user query and two responses, produce a comprehensive and well-structured set (ten) of evaluation rubric that can be used to distinguish the relative quality of the two responses.\nProvide a list of candidate rubrics ordered by importance from highest to lowest. These rubrics should be sufficiently discriminative to accurately capture meaningful differences between the responses.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output:"""

direct_judge_prompt = """You are an objective, impartial, and unbiased content evaluator. Provide a single final verdict indicating which response best fulfills the user’s intent.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\n\n\nRequired output format:\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your final verdict:"""

judge_prompt="""You are an objective, impartial, and unbiased content evaluator.\nBased on predefined list of evaluation rubrics, determine the most critical evaluation dimensions for assessing alignment with the user’s intent. Perform a rigorous, evidence-based comparison across these dimensions and deliver a single final verdict identifying the response that best fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric-Judge> Rubric Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B>\nThe final verdict is [[A]] or [[B]]</Rubric-Judge>\n\n\n###Please output your analysis and final verdict:"""

select_rubric_prompt="""Given a predefined set of evaluation rubrics, identify the rubric that is most critical for assessing alignment with the user’s intent.### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric> Name. Explanation.</Rubric>\n\n\n###Please output your analysis:"""

judge_rubric_prompt="""You are an objective, impartial, and unbiased content evaluator.\nBased on the provided evaluation rubric, conduct a rigorous, evidence-based comparison of two responses and provide a single final verdict indicating which response best fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric-Judge> Rubric Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B>\nThe final verdict is [[A]] or [[B]]</Rubric-Judge>\n\n\n###Please output your analysis and final verdict:"""

correction_prompt = """Using the evaluator’s judge as authoritative guidance, revise the response accordingly. All modifications must strictly follow the evaluator’s feedback and target the specific deficiencies identified. Avoid minor edits; each issue should be addressed independently with substantive revisions that fully resolve the evaluator’s concerns.### Required output format (produce exactly this structure — replace placeholders with real content):\n<Revised Response>xxx</Revised Response>\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant Answer]:\n{response}\n[The End of Assistant Answer]\n\n[The Start of Comments]:\n{criteria}\n[The End of  Comments]\n\n\n###Output only the improved versions of Response:"""




def calculate_judge_score(content, answer):

    def extract_omni_answer(response: str, answer: int):
        
        assert type(answer) == int, f"answer shoule be a interger. not [{type(answer)}]"
        assert answer in [0, 1], f"answer shoule be 0 or 1, not [{answer}]"

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

    correct, pred = extract_omni_answer(content, answer)
    if correct:
        return 1.0, pred
    return 0.0, pred


def parse_qa_string(text):
    # 注意：原文中有转义的单引号 A\'s → 实际字符串中可能是 "A's" 或保留反斜杠
    # 我们用正则匹配 [...] 标记（允许内部有空格、引号、转义等），非贪婪捕获内容
    pattern = r'''
        \[User\ Question\]\s*(.*?)\s*
        \[The\ Start\ of\ Assistant\ A\\?'s\ Answer\]\s*(.*?)\s*
        \[The\ End\ of\ Assistant\ A\\?'s\ Answer\]
        \s*
        \[The\ Start\ of\ Assistant\ B\\?'s\ Answer\]\s*(.*?)\s*
        \[The\ End\ of\ Assistant\ B\\?'s\ Answer\]
    '''
    # 使用 re.DOTALL 使 . 匹配换行符，re.VERBOSE 便于写多行 pattern
    match = re.search(pattern, text, re.DOTALL | re.VERBOSE | re.IGNORECASE)
    
    if match:
        query = match.group(1).strip()
        response1 = match.group(2).strip()
        response2 = match.group(3).strip()
        return {
            'query': query,
            'response1': response1,
            'response2': response2
        }
    else:
        raise ValueError("Failed to parse the input string. Check format.")


def parse_rubric_judge_output(output: str, single: bool=False):
    
    results = {"rubric_judge": [], "verdict": None}
    
    if single:
        criteria_matches = re.findall(r"<criteria>(.*?)</criteria>", output, re.S)
    else:
        criteria_matches = re.findall(r"<criteria\s*\d+>(.*?)</criteria\s*\d+>", output, re.S)
    
    for crit in criteria_matches:
        crit_before_A = crit.split("<Judge A>")[0].strip()
        # 提取 Judge A
        judge_A_match = re.search(r"<Judge A>(.*?)</Judge A>", crit, re.S)
        judge_A = judge_A_match.group(1).strip() if judge_A_match else None

        # 提取 Judge B
        judge_B_match = re.search(r"<Judge B>(.*?)</Judge B>", crit, re.S)
        judge_B = judge_B_match.group(1).strip() if judge_B_match else None
        
        criterion_dict={
            "criterion": crit_before_A,
            "judge_A": judge_A,
            "judge_B": judge_B,
        }

        if criterion_dict["criterion"] is None or criterion_dict['judge_A'] is None or criterion_dict['judge_B'] is None:
            raise Exception(f"Parse Criterion failed. Text:[{crit}]. will skip it.")
        
        results["rubric_judge"].append(criterion_dict)
    assert len(results["rubric_judge"]) > 0

    pred = None
    pattern = r'\[{1,2}([abcABC])\]{1,2}'
    matches = re.findall(pattern, output)
    if len(matches) > 0:
        pred = matches[-1]
    else:
        match = re.search(r"<Final Verdict>(.*?)</Final Verdict>", output)
        pred = match.group(1) if match else None
    
    if isinstance(pred, str):
        pred = pred.upper()
    results["verdict"] = pred

    return results


query2data = {}
dpo_data = []
for line in open("/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/model_scope/data/grm_lang/grm_lang_dpo_mix.jsonl").readlines():

    json_item = json.loads(line)
    result = parse_qa_string(json_item['conversations'][0]['content'])

    query = result['query'].replace(":\n", "").strip()
    response1 = result['response1'].replace(":\n", "").strip()
    response2 = result['response2'].replace(":\n", "").strip()
    
    chosen_judge = parse_rubric_judge_output(json_item['chosen']['content'])
    rejected_judge = parse_rubric_judge_output(json_item['rejected']['content'])
    
    query2data[query] = {
        "response1": response1,
        "response2": response2,
        "chosen_judge": chosen_judge,
        "rejected_judge": rejected_judge
    }
    # import pdb; pdb.set_trace()


for i in range(10):
            
    id2data = {}
    
    criteria_file = "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/model_scope/0919_results/language_77k-0{}/criteria.jsonl".format(i)
    judge_file = "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/model_scope/0919_results/language_77k-0{}/judge.jsonl".format(i)
    ranking_file = "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/model_scope/0919_results/language_77k-0{}/ranking.jsonl".format(i)
    refinement_file = "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/model_scope/0919_results/language_77k-0{}/refinment.jsonl".format(i)

    for line in open(criteria_file).readlines():
        json_item = json.loads(line)
        idx = json_item['paired_data']['id']
        id2data[idx] = {
            "query": json_item['paired_data']['query'],
            "chosen": json_item['paired_data']['chosen'],
            "rejected": json_item['paired_data']['rejected'],
            "criteria_list": json_item['criteria_list'],
            "answer": json_item['answer']
        }

    for line in open(judge_file).readlines():
        json_item = json.loads(line)
        idx = json_item['paired_data']['id']
        if idx in id2data:
            id2data[idx]['judge_pair'] = json_item['judge_pair']
            id2data[idx]['judge'] = json_item['judge']

    for line in open(ranking_file).readlines():
        json_item = json.loads(line)
        idx = json_item['paired_data']['id']
        if 'ranking_pair' in json_item and idx in id2data:
            id2data[idx]['ranking_pair'] = json_item['ranking_pair']

    for line in open(refinement_file).readlines():
        json_item = json.loads(line)
        idx = json_item['paired_data']['id']
        if idx in id2data:
            id2data[idx]['refinement_list'] = json_item['refinement_list']

    correct = 0
    correct_count = 0
    print(len(id2data))
    for key, value in id2data.items():

        if "ranking_pair" not in value:
            continue
        
        s_meta = []
        r1 = value['ranking_pair']['ranking_raw_a']
        r2 = value['ranking_pair']['ranking_raw_b']
        for k, s1, s2 in zip(range(10), value['ranking_pair']['ranking_a'], value['ranking_pair']['ranking_b']):
            
            _, pred = calculate_judge_score(value['judge'][k], value['answer'])
            
            if pred == "A" and s1 > s2:
                s_meta.append(s2 - r2)
            elif pred == "A" and s1 < s2:
                s_meta.append(s1 - s2)
            elif pred == "B" and s1 < s2:
                s_meta.append(s1 - r1)
            elif pred == "B" and s1 > s2:
                s_meta.append(s2 - s1)
            
        
        sorted_indexed = [idx for idx, val in sorted(list(enumerate(s_meta)), key=lambda x: x[1], reverse=True)]
        
        criteria_list = []
        for c in value["criteria_list"]:
            criteria_list.append(c['content'])
        
        if len(criteria_list) != 10 or len(sorted_indexed) != 10:
            continue

        if value['answer'] == 0:
            response_1 = value['chosen']['content']
            response_2 = value['rejected']['content']
        elif value['answer'] == 1:
            response_2 = value['chosen']['content']
            response_1 = value['rejected']['content']
        else:
            continue
        
        if s_meta[0] - s_meta[-1] < 20:
            continue

        chosen_idx = sorted_indexed[0]
        rejected_idx = sorted_indexed[-1]
        chosen_judge, chosen_pred = calculate_judge_score(value['judge'][chosen_idx], value['answer'])
        rejected_judge, rejected_pred = calculate_judge_score(value['judge'][rejected_idx], value['answer'])

        order_criteria_list = []
        for index in sorted_indexed:
            order_criteria_list.append(criteria_list[index])

        if chosen_judge and not rejected_judge:
            chosen_rubric = criteria_list[chosen_idx]
            rejected_rubric = criteria_list[rejected_idx]

            data_rubric_planning = {
                "conversations": [
                    {"role": "user", "content": rubric_prompt.format(query=value['query']['content'], response_1=response_1,response_2=response_2)}
                ],
                "chosen": {"role": "assistant", "content": "\n".join(order_criteria_list).strip()},
                "rejected": {"role": "assistant", "content": "\n".join(criteria_list).strip()},
                "images": []
            }

            chosen_judge = f"<Rubric-Judge>{criteria_list[chosen_idx].strip()}\n<Judge A>{value['judge_pair']['judge_a_list'][chosen_idx].strip()}</Judge A>\n<Judge B>{value['judge_pair']['judge_b_list'][chosen_idx].strip()}</Judge B>\nThe Final Verdict is [[{chosen_pred}]].</Rubric-Judge>"
            
            rejected_judge = f"<Rubric-Judge>{criteria_list[rejected_idx].strip()}\n<Judge A>{value['judge_pair']['judge_a_list'][rejected_idx].strip()}</Judge A>\n<Judge B>{value['judge_pair']['judge_b_list'][rejected_idx].strip()}</Judge B>\nThe Final Verdict is [[{rejected_pred}]].</Rubric-Judge>"

            data_rubric_judge = {
                "conversations": [
                    {"role": "user", "content": rubric_prompt.format(query=value['query']['content'],response_1=response_1,response_2=response_2)},
                    {"role": "assistant", "content": "\n".join(criteria_list).strip()},
                    {"role": "user", "content": judge_prompt},
                ],
                "chosen": {"role": "assistant", "content": chosen_judge},
                "rejected": {"role": "assistant", "content": rejected_judge},
                "images": []
            }

            dpo_data.append(data_rubric_planning)
            dpo_data.append(data_rubric_judge)

            
            if value['query']['content'] in query2data:
                
                _query = value['query']['content']
                _item = query2data[value['query']['content']]
                _response_1 = _item['response1']
                _response_2 = _item['response2']
                _chosen_judge = _item['chosen_judge']
                _rejected_judge = _item['rejected_judge']

                for i in range(3):

                    _criteria_list = criteria_list.copy()
                    _criteria_list[random.randint(0, 5)] = _chosen_judge['rubric_judge'][i]['criterion']
                    _criteria_list[random.randint(5, 9)] = _rejected_judge['rubric_judge'][i]['criterion']
                    
                    m_chosen_judge = f"<Rubric-Judge>{_chosen_judge['rubric_judge'][i]['criterion']}\n<Judge A>{_chosen_judge['rubric_judge'][i]['judge_A'].strip()}</Judge A>\n<Judge B>{_chosen_judge['rubric_judge'][i]['judge_B'].strip()}</Judge B>\nThe Final Verdict is [[{_chosen_judge['verdict']}]].</Rubric-Judge>"
                    
                    m_rejected_judge = f"<Rubric-Judge>{_rejected_judge['rubric_judge'][i]['criterion']}\n<Judge A>{_rejected_judge['rubric_judge'][i]['judge_A'].strip()}</Judge A>\n<Judge B>{_rejected_judge['rubric_judge'][i]['judge_B'].strip()}</Judge B>\nThe Final Verdict is [[{_rejected_judge['verdict']}]].</Rubric-Judge>"

                    m_data_rubric_judge = {
                        "conversations": [
                            {"role": "user", "content": rubric_prompt.format(query=value['query']['content'],response_1=response_1,response_2=response_2)},
                            {"role": "assistant", "content": "\n".join(_criteria_list).strip()},
                            {"role": "user", "content": judge_prompt},
                        ],
                        "chosen": {"role": "assistant", "content": m_chosen_judge},
                        "rejected": {"role": "assistant", "content": m_rejected_judge},
                        "images": []
                    }
                
                    dpo_data.append(m_data_rubric_judge)

random.shuffle(dpo_data)
print(len(dpo_data))
with open("/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/dual_rm_lang_dpo.jsonl", 'w') as fw:
    for line in dpo_data:
            
        if "<video>" in str(line) or "<image>" in str(line) or "<audio>" in str(line):
            continue
    
        fw.writelines(json.dumps(line) + "\n")
