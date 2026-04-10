import json
import re
import random
import pandas as pd
import Levenshtein
import numpy as np
import os


rubric_prompt = """Given a user query and two responses, produce a comprehensive and well-structured set (ten) of evaluation rubric that can be used to distinguish the relative quality of the two responses.\nProvide a list of candidate rubrics ordered by importance from highest to lowest. These rubrics should be sufficiently discriminative to accurately capture meaningful differences between the responses.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output:"""

direct_judge_prompt = """You are an objective, impartial, and unbiased content evaluator. Provide a single final verdict indicating which response best fulfills the user’s intent.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\n\n\nRequired output format:\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your final verdict:"""

judge_prompt="""You are an objective, impartial, and unbiased content evaluator.\nBased on predefined list of evaluation rubrics, determine the most critical evaluation dimensions for assessing alignment with the user’s intent. Perform a rigorous, evidence-based comparison across these dimensions and deliver a single final verdict identifying the response that best fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric-Judge> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B>\nThe final verdict is [[A]] or [[B]]</Rubric-Judge>\n\n\n###Please output your analysis and final verdict:"""

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

def parse(text):
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return  clean_text


def prepare_language_sft(sft_data):

    id2omni = {}
    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Skywork-Reward-Preference/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-UltraFeedback/train-00000-of-00001.parquet"]:
        
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            assert len(row['conversations']) == 2
            golden_rubric = row['conversations'][0]['value']
            query = row['conversations'][1]['value']
            chosen = row['chosen']['value']
            rejected = row['rejected']['value']
            id2omni[query] = row
    hit = 0
    count = 0
    for i in range(10):
        
        id2data = {}
        
        criteria_file = "/workspace/mnt/lxb_work/MindMirror/model_scope/0919_results/language_77k-0{}/criteria.jsonl".format(i)
        judge_file = "/workspace/mnt/lxb_work/MindMirror/model_scope/0919_results/language_77k-0{}/judge.jsonl".format(i)
        ranking_file = "/workspace/mnt/lxb_work/MindMirror/model_scope/0919_results/language_77k-0{}/ranking.jsonl".format(i)
        refinement_file = "/workspace/mnt/lxb_work/MindMirror/model_scope/0919_results/language_77k-0{}/refinment.jsonl".format(i)

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
        for key, value in id2data.items():
            
            if 'ranking_pair' not in value or 'ranking_pair' not in value:
                continue
            
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
            
            if len(value["criteria_list"]) != 10:
                continue
            
            criteria_list = []
            for idx, c in enumerate(value["criteria_list"]):
                criteria_list.append(c['content'])

            selected_criteria = []
            selected_criteria_p1 = []
            selected_criteria_a = []
            selected_criteria_b = []
            revised_response_a = []
            revised_response_b = []
            order_criteria_list = []
            for index in sorted_indexed:
                
                order_criteria_list.append(criteria_list[index])
                judge, pred = calculate_judge_score(value['judge'][index], value['answer'])
                
                if judge and s_meta[index] > 5:
                    
                    selected_item = f"<Rubric-Judge>{criteria_list[index].strip()}\n<Judge A>{value['judge_pair']['judge_a_list'][index].strip()}</Judge A>\n<Judge B>{value['judge_pair']['judge_b_list'][index].strip()}</Judge B>"
                    selected_criteria.append(selected_item)
                    
                    selected_criteria_p1.append(f"<Rubric>{criteria_list[index].strip()}</Rubric>")

                    selected_criteria_a.append(f"{criteria_list[index].strip()}\n<Judge>{value['judge_pair']['judge_a_list'][index].strip()}</Judge>")

                    selected_criteria_b.append(f"{criteria_list[index].strip()}\n<Judge>{value['judge_pair']['judge_b_list'][index].strip()}</Judge>")
                    
                    revised_response_a.append(f"<Revised Response>{parse(value['refinement_list']['refinement_a'][index]).strip()}</Revised Response>")
                    revised_response_b.append(f"<Revised Response>{parse(value['refinement_list']['refinement_b'][index]).strip()}</Revised Response>")
            
            for k in range(len(selected_criteria)):
                
                if k == 0:

                    if value['answer'] == 0:
                        response_1 = value['chosen']['content']
                        response_2 = value['rejected']['content']
                        judge_response = selected_criteria[k] + "\n" + "The Final Verdict is [[A]].</Rubric-Judge>"
                    elif value['answer'] == 1:
                        response_2 = value['chosen']['content']
                        response_1 = value['rejected']['content']
                        judge_response = selected_criteria[k] + "\n" + "The Final Verdict is [[B]].</Rubric-Judge>"
                    else:
                        continue
                    
                    if "<video>" in str(response_1) or "<image>" in str(response_1) or "<audio>" in str(response_1):
                        continue
                    if "<video>" in str(response_2) or "<image>" in str(response_2) or "<audio>" in str(response_2):
                        continue
                    
                    if value['query']['content'] in id2omni.keys():
                        hit += 1
                        rubric = id2omni[value['query']['content']]['conversations'][0]['value']
                        new_order_criteria_list = criteria_list.copy()
                        new_order_criteria_list[random.randint(0,9)] = rubric
                        omni_data = {
                            "conversations": [
                                {"role": "user", "content": rubric_prompt.format(query=value['query']['content'],response_1=response_1,response_2=response_2)},
                                {"role": "assistant", "content": "\n".join(new_order_criteria_list).strip()},
                                {"role": "user", "content": select_rubric_prompt},
                                {"role": "assistant", "content": f"<Rubric>{rubric}</Rubric>"}
                            ],
                            "images": []
                        }
                        sft_data.append(omni_data)
                    data = {
                        "conversations": [
                            {"role": "user", "content": rubric_prompt.format(query=value['query']['content'],response_1=response_1,response_2=response_2)},
                            {"role": "assistant", "content": "\n".join(order_criteria_list).strip()},
                            {"role": "user", "content": judge_prompt},
                            {"role": "assistant", "content": judge_response}
                        ],
                        "images": []
                    }
                    data_v2 = {
                        "conversations": [
                            {"role": "user", "content": rubric_prompt.format(query=value['query']['content'],response_1=response_1,response_2=response_2)},
                            {"role": "assistant", "content": "\n".join(order_criteria_list).strip()},
                            {"role": "user", "content": select_rubric_prompt},
                            {"role": "assistant", "content": selected_criteria_p1[k]},
                            {"role": "user", "content": judge_rubric_prompt},
                            {"role": "assistant", "content": judge_response}
                        ],
                        "images": []
                    }
                    if "<video>" in str(data) or "<image>" in str(data) or "<audio>" in str(data):
                        continue
                    
                    if random.random() > 0.9:
                        sft_data.append(data)
                    else:
                        sft_data.append(data_v2)
                    
                    if value['answer'] == 0:
                        _revised_response_b = revised_response_b[k].replace("<Revised Response>", "").replace("</Revised Response>", "")
                        dist = Levenshtein.distance(response_2, _revised_response_b)
                        mod_ratio = dist / max(len(response_2), 1)
                        if mod_ratio > 0.5:
                            data_2 = {
                                "conversations": [
                                    {"role": "user", "content": correction_prompt.format(query=value['query']['content'],response=response_2,criteria=selected_criteria_b[k])},
                                    {"role": "assistant", "content": revised_response_b[k]}
                                ],
                                "images": []
                            }
                            if "<video>" in str(data_2) or "<image>" in str(data_2) or "<audio>" in str(data_2):
                                continue
                            if random.random() > 0.5:
                                sft_data.append(data_2)
                    else:
                        _revised_response_a = revised_response_a[k].replace("<Revised Response>", "").replace("</Revised Response>", "")
                        dist = Levenshtein.distance(response_1, _revised_response_a)
                        mod_ratio = dist / max(len(response_1), 1)
                        if mod_ratio > 0.5:
                            data_2 = {
                                "conversations": [
                                    {"role": "user", "content": correction_prompt.format(query=value['query']['content'],response=response_1,criteria=selected_criteria_a[k])},
                                    {"role": "assistant", "content": revised_response_a[k]}
                                ],
                                "images": []
                            }
                            if "<video>" in str(data_2) or "<image>" in str(data_2) or "<audio>" in str(data_2):
                                continue

                            if random.random() > 0.5:
                                sft_data.append(data_2)
                # print("language_sft: ", len(sft_data), hit)

    print("language_sft: ", len(sft_data))



def prepare_t2i_sft(sft_data):

    dir = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/"
    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-HPDv2/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/EvalMuse/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Open-Image-Preferences/train-00000-of-00001.parquet"]:

        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            
            prompt = direct_judge_prompt.format(query=row['chosen']['value'],response_1="<image>",response_2="<image>")
            
            if random.random() > 0.5:
                image_1 = (dir + row['images'][0]).replace("omni_t2i/", "").replace("omni_t2i_hpd/", "").replace("evalmuse/", "")
                assert os.path.exists(image_1), image_1
                image_2 = (dir + row['images'][1]).replace("omni_t2i/", "").replace("omni_t2i_hpd/", "").replace("evalmuse/", "")
                assert os.path.exists(image_2)
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[A]]"}
                    ],
                    "images": [image_1, image_2]
                }
            else:
                image_1 = (dir + row['images'][1]).replace("omni_t2i/", "").replace("omni_t2i_hpd/", "").replace("evalmuse/", "")
                assert os.path.exists(image_1), image_1
                image_2 = (dir + row['images'][0]).replace("omni_t2i/", "").replace("omni_t2i_hpd/", "").replace("evalmuse/", "")
                assert os.path.exists(image_2)
                
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[B]]"}
                    ],
                    "images": [image_1, image_2]
                }
            sft_data.append(data)
    
    print("vision_t2i_sft: ", len(sft_data))


def prepare_ti2t_sft(sft_data):

    dir = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/"
    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-RLAIF-V/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/RLAIF-V/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/OmniAlign-V-DPO/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-VLFeedback/train-00000-of-00001.parquet"]:

        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            
            if random.random() > 0.1:
                continue
            
            assert len(row['images']) == 1
            image_1 = (dir + row['images'][0]).replace("omni_ti2t/", "").replace("rlaifv/", "").replace("omnialign/", "")
            assert os.path.exists(image_1), image_1

            if random.random() > 0.5:
                prompt = direct_judge_prompt.format(query=row['conversations'][-1]['value'],response_1=row['chosen']['value'],response_2=row['rejected']['value'])
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[A]]"}
                    ],
                    "images": [image_1]
                }
            else:
                prompt = direct_judge_prompt.format(query=row['conversations'][-1]['value'],response_1=row['rejected']['value'],response_2=row['chosen']['value'])
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[B]]"}
                    ],
                    "images": [image_1]
                }
            sft_data.append(data)
    
    print("vision_t2i_sft: ", len(sft_data))


def prepare_t2v_sft(sft_data):

    dir = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/"
    # image_1 = (dir + row['images'][0]).replace("omni_ti2t/", "").replace("rlaifv/", "").replace("omnialign/", "")
    # assert os.path.exists(image_1), image_1

    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VideoDPO/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VisionRewardDB-Video/train-00000-of-00001.parquet"]:
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            
            for r in row['images']:
                assert os.path.exists(dir + r.replace("videodpo_image", "").replace("visionrewardv", "")), dir + r

            if random.random() > 0.5:
                prompt = direct_judge_prompt.format(query=row['chosen']['value'],response_1=row['conversations'][-1]['value'],response_2=row['conversations'][-1]['value'])
                
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[A]]"}
                    ],
                    "images": [dir + r.replace("videodpo_image/", "").replace("visionrewardv/", "") for r in row['images']]
                }
            else:
                prompt = direct_judge_prompt.format(query=row['chosen']['value'],response_1=row['conversations'][-1]['value'],response_2=row['conversations'][-1]['value'])
                data = {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "The final verdict is [[B]]"}
                    ],
                    "images": [dir + r.replace("videodpo_image/", "").replace("visionrewardv/", "") for r in np.concatenate((row['images'][len(row['images'])//2:],row['images'][:len(row['images'])//2]))]
                }
            sft_data.append(data)
    
    print("prepare_tiv_sft: ", len(sft_data))


def data_stat():

    # Language
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Skywork-Reward-Preference/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Skywork-Reward-Preference/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-UltraFeedback/train-00000-of-00001.parquet

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Skywork-Reward-Preference/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Skywork-Reward-Preference len={count}")
    
    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Skywork-Reward-Preference/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-Skywork-Reward-Preferenc len={count}")
    
    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-UltraFeedback/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-UltraFeedback len={count}")   

    # Vision Text to Image
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/HPDv2/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/EvalMuse/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-HPDv2/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Open-Image-Preferences/train-00000-of-00001.parquet

    # import pdb; pdb.set_trace()
    # path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/HPDv2/train-00000-of-00001.parquet"
    # df = pd.read_parquet(path)
    # count = 0
    # for idx, row in df.iterrows():
    #     count += 1
    # print(f"HPDv2 len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/EvalMuse/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"EvalMuse len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-HPDv2/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-HPDv2 len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Open-Image-Preferences/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-Open-Image-Preferences len={count}")

    # Vision Text Image to Text
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-RLAIF-V/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/RLAIF-V/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/OmniAlign-V-DPO/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-VLFeedback/train-00000-of-00001.parquet

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-RLAIF-V/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-RLAIF-V len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/RLAIF-V/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"RLAIF-V len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/OmniAlign-V-DPO/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"OmniAlign-V-DPO len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-VLFeedback/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"Omni-VLFeedback len={count}")

    # T2V
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VideoDPO/train-00000-of-00001.parquet
    # /workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VisionRewardDB-Video/train-00000-of-00001.parquet

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VideoDPO/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"VideoDPO len={count}")

    path = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/VisionRewardDB-Video/train-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    count = 0
    for idx, row in df.iterrows():
        count += 1
    print(f"VisionRewardDB-Video len={count}")

    # 采样 分为两个数据集 SFT Train and RL
    # Omni-Skywork-Reward-Preference len=16376
    # Omni-UltraFeedback len=7901

    # EvalMuse len=2944
    # Omni-HPDv2 len=8959
    # Omni-Open-Image-Preferences len=8105

    # Omni-RLAIF-V len=15867
    # RLAIF-V len=83124
    # OmniAlign-V-DPO len=133341
    # Omni-VLFeedback len=12311

    # VideoDPO len=10000
    # VisionRewardDB-Video len=1795

def prepare_omni_sft():

    # data_stat()
    sft_data = []
    rl_data = []
    
    print("start...")
    

    # prepare_t2i_sft(sft_data)
    # prepare_ti2t_sft(sft_data)
    # prepare_t2v_sft(sft_data)
    
    # print(len(sft_data))
    # random.shuffle(sft_data)
    # sft_data = sft_data[:10000]
    
    # print(len(sft_data))
    prepare_language_sft(sft_data)
    
    print(len(sft_data))
    random.shuffle(sft_data)
    # with open("/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/dual_rm_omni_sft.jsonl", "w") as f:
    with open("/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/dual_rm_lang_sft_v2.jsonl", "w") as f:
        for d in sft_data:
            f.writelines(json.dumps(d) + "\n")

prepare_omni_sft()
