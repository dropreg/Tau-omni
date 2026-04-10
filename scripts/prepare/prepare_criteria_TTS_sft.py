import json
import re

criteria_prompt = """Given a user query and two responses, produce a comprehensive and well-structured set of evaluation criteria that can be used to distinguish the relative quality of the two responses. The evaluation criteria should be formulated in a way that is directly applicable to human preference annotation or reward-model training.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output:"""

judge_prompt="""You are an objective, impartial, and unbiased content evaluator.
Given a criteria list describing how two responses should be compared, identify the most critical evaluation dimensions that are most relevant to determining which response better fulfills the user’s intent. Then produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Criteria> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria>\n...\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your analysis and final verdict:"""

judge_p1_prompt="""You are an objective, impartial, and unbiased content evaluator.
Given a criteria list describing how two responses should be compared, identify the most critical evaluation dimensions that are most relevant to determining which response better fulfills the user’s intent. Then produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Criteria> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria>\n...\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your analysis and final verdict:"""

judge_p2_prompt="""You are an objective, impartial, and unbiased content evaluator.
Given a criteria list describing how two responses should be compared, identify the three most critical evaluation dimensions that are most relevant to determining which response better fulfills the user’s intent. Then produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Criteria> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria>\n...\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your analysis and final verdict:"""

correction_prompt = """Based on the evaluator’s comments, revise both Response A and Response B. Your revisions must strictly follow the evaluator’s feedback. Do not simply merge the two responses; modify each independently based on its respective issues. \n\n\n###Output only the improved versions of Response A and Response B:"""

correction_v2_prompt = """Based on the evaluator’s comments, revise Response. Your revisions must strictly follow the evaluator’s feedback. Do not simply merge the two responses; modify each independently based on its respective issues. \n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant Answer]:\n{response}\n[The End of Assistant Answer]\n\n[The Start of Comments]:\n{criteria}\n[The End of  Comments]\n\n\n###Output only the improved versions of Response"""

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

def parse(text):
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return  clean_text

def extract():
    count = 0
    with open("/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/dual_rm_skywork_sft.jsonl", "w") as f:
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
                try:
                
                    s_meta = []
                    r1 = value['ranking_pair']['ranking_raw_a']
                    r2 = value['ranking_pair']['ranking_raw_b']
                    for k, s1, s2 in zip(range(10), value['ranking_pair']['ranking_a'], value['ranking_pair']['ranking_b']):
                        
                        _, pred = calculate_judge_score(value['judge'][k], value['answer'])
                        
                        if pred == "A" and s1 > s2:
                            s_meta.append(s2 - r2)
                            # s_meta.append(s2 + s1 - r1 - r2)
                        elif pred == "A" and s1 < s2:
                            s_meta.append(s1 - s2)
                            # s_meta.append(s2 + s1 - r1 - r2)
                        elif pred == "B" and s1 < s2:
                            s_meta.append(s1 - r1)
                            # s_meta.append(s2 + s1 - r1 - r2)
                        elif pred == "B" and s1 > s2:
                            s_meta.append(s2 - s1)
                            # s_meta.append(s2 + s1 - r1 - r2)

                    sorted_indexed = [idx for idx, val in sorted(list(enumerate(s_meta)), key=lambda x: x[1], reverse=True)]
                    
                    # if calculate_judge_score(value['judge'][sorted_indexed[j]], value['answer'])[0]:
                    #     correct += 1
                    # correct_count += 1

                    criteria_list = []
                    for c in value["criteria_list"]:
                        criteria_list.append(c['content'])

                    selected_criteria = []
                    selected_criteria_a = []
                    selected_criteria_b = []
                    revised_response_a = []
                    revised_response_b = []
                    for index in sorted_indexed:

                        if not (s_meta[index] > r1 and s_meta[index] > r2):
                            continue

                        judge, pred = calculate_judge_score(value['judge'][index], value['answer'])
                        
                        if judge:
                            selected_item = f"<Criteria {len(selected_criteria) + 1}>{value['criteria_list'][index]['content'].strip()}\n<Judge A>{value['judge_pair']['judge_a_list'][index].strip()}</Judge A>\n<Judge B>{value['judge_pair']['judge_b_list'][index].strip()}</Judge B></Criteria {len(selected_criteria) + 1}>"
                            selected_criteria.append(selected_item)

                            selected_criteria_a.append(f"<Criteria {len(selected_criteria) + 1}>{value['criteria_list'][index]['content'].strip()}\n<Judge>{value['judge_pair']['judge_a_list'][index].strip()}</Judge></Criteria {len(selected_criteria) + 1}>")
                            selected_criteria_b.append(f"<Criteria {len(selected_criteria) + 1}>{value['criteria_list'][index]['content'].strip()}\n<Judge>{value['judge_pair']['judge_b_list'][index].strip()}</Judge></Criteria {len(selected_criteria) + 1}>")
                            
                            revised_response_a.append(f"<Response>{parse(value['refinement_list']['refinement_a'][index]).strip()}</Response>")
                            revised_response_b.append(f"<Response>{parse(value['refinement_list']['refinement_b'][index]).strip()}</Response>")

                        if len(selected_criteria) >= 3:
                            break
                    
                    if len(selected_criteria) < 3:
                        continue
                    
                    if value['answer'] == 0:
                        response_1 = value['chosen']
                        response_2 = value['rejected']
                        judge_response = "\n".join(selected_criteria) + "\n" + "The Final Verdict is [[A]]"
                    elif value['answer'] == 1:
                        response_2 = value['chosen']
                        response_1 = value['rejected']
                        judge_response = "\n".join(selected_criteria) + "\n" + "The Final Verdict is [[B]]"
                    else:
                        continue
                    
                    data = {
                        "conversations": [
                            {"role": "user", "content": criteria_prompt.format(query=value['query'],response_1=response_1,response_2=response_2)},
                            {"role": "assistant", "content": "\n".join(criteria_list).strip()},
                            {"role": "user", "content": judge_prompt},
                            {"role": "assistant", "content": judge_response}
                        ],
                        "images": []
                    }
                    
                    if "<video>" not in str(data) and "<image>" not in str(data) and "<audio>" not in str(data):
                        count += 1
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

                    if value['answer'] == 0:
                        data_2 = {
                            "conversations": [
                                {"role": "user", "content": correction_v2_prompt.format(query=value['query'],response=response_2,criteria=selected_criteria_a[0])},
                                {"role": "assistant", "content": revised_response_b[0]}
                            ],
                            "images": []
                        }
                    else:
                        data_2 = {
                            "conversations": [
                                {"role": "user", "content": correction_v2_prompt.format(query=value['query'],response=response_1,criteria=selected_criteria_b[0])},
                                {"role": "assistant", "content": revised_response_a[0]}
                            ],
                            "images": []
                        }
                    if "<video>" not in str(data_2) and "<image>" not in str(data_2) and "<audio>" not in str(data_2):

                        f.write(json.dumps(data_2, ensure_ascii=False) + "\n")
                except:
                    continue
    print(count)

extract()

