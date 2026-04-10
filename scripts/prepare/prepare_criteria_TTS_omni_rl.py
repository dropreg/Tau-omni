import json
import re
import random
import pandas as pd
import Levenshtein
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq

rubric_prompt = """Given a user query and two responses, produce a comprehensive and well-structured set (ten) of evaluation rubric that can be used to distinguish the relative quality of the two responses.\nProvide a list of candidate rubrics ordered by importance from highest to lowest. These rubrics should be sufficiently discriminative to accurately capture meaningful differences between the responses.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output:"""

direct_judge_prompt = """You are an objective, impartial, and unbiased content evaluator. Provide a single final verdict indicating which response best fulfills the user’s intent.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\n\n\nRequired output format:\nThe final verdict is [[A]] or [[B]]\n\n\n###Please output your final verdict:"""

judge_prompt="""You are an objective, impartial, and unbiased content evaluator.\nBased on predefined list of evaluation rubrics, determine the most critical evaluation dimensions for assessing alignment with the user’s intent. Perform a rigorous, evidence-based comparison across these dimensions and deliver a single final verdict identifying the response that best fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric-Judge> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B>\nThe final verdict is [[A]] or [[B]]</Rubric-Judge>\n\n\n###Please output your analysis and final verdict:"""

select_rubric_prompt="""Given a predefined set of evaluation rubrics, identify the rubric that is most critical for assessing alignment with the user’s intent.### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric> Name. Explanation.</Rubric>\n\n\n###Please output your analysis:"""

judge_rubric_prompt="""You are an objective, impartial, and unbiased content evaluator.\nBased on the provided evaluation rubric, conduct a rigorous, evidence-based comparison of two responses and provide a single final verdict indicating which response best fulfills the user’s intent.\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Rubric-Judge> Rubric Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B>\nThe final verdict is [[A]] or [[B]]</Rubric-Judge>\n\n\n###Please output your analysis and final verdict:"""

correction_prompt = """Using the evaluator’s judge as authoritative guidance, revise the response accordingly. All modifications must strictly follow the evaluator’s feedback and target the specific deficiencies identified. Avoid minor edits; each issue should be addressed independently with substantive revisions that fully resolve the evaluator’s concerns.### Required output format (produce exactly this structure — replace placeholders with real content):\n<Revised Response>xxx</Revised Response>\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant Answer]:\n{response}\n[The End of Assistant Answer]\n\n[The Start of Comments]:\n{criteria}\n[The End of  Comments]\n\n\n###Output only the improved versions of Response:"""


def build_criteria_TTS_conversation(idx, query, response_1, response_2, chosen, rejected, ground_truth, golden_rubric):

    prompt = rubric_prompt.format(query=query, response_1=response_1, response_2=response_2)
    
    data = {
        "data_source": "Omni_Reward_Data",
        "agent_name": "criteria_TTS",
        "prompt": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
        'reward_model': {
            'ground_truth': ground_truth,
        },
        "extra_info": {
            "idx":  idx,
            "interaction_kwargs": {
                "name": "criteria_TTS_interaction",
                "mode": "meta_reward_golden",
                "response_1": response_1,
                "response_2": response_2,
                "query": query,
                "chosen": chosen,
                "rejected": rejected,
                "golden_rubric": golden_rubric,
                "ground_truth": ground_truth
            },
        }
    }
    
    return data

def prepare_language_sft(rl_data):

    count = 0
    dir = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/"
    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Skywork-Reward-Preference/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-UltraFeedback/train-00000-of-00001.parquet"]:
        
        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            count += 1
            assert len(row['conversations']) == 2
            golden_rubric = row['conversations'][0]['value']
            query = row['conversations'][1]['value']
            chosen = row['chosen']['value']
            rejected = row['rejected']['value']

            if random.random() > 0.5:
                ground_truth = 0
                response_1 = chosen
                response_2 = rejected
            else:
                ground_truth = 1
                response_1 = rejected
                response_2 = chosen

            data = build_criteria_TTS_conversation("omni_reward_data_" + str(count), query, response_1, response_2, chosen, rejected, ground_truth, golden_rubric)
            rl_data.append(data)

    print(count, len(rl_data))

def prepare_t2i_sft(sft_data):

    dir = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/"
    for path in ["/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-HPDv2/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/EvalMuse/train-00000-of-00001.parquet", "/workspace/mnt/lxb_work/hf_dir/hf_dataset/OmniRewardData/Omni-Open-Image-Preferences/train-00000-of-00001.parquet"]:

        df = pd.read_parquet(path)
        for idx, row in df.iterrows():
            
            data = build_criteria_TTS_conversation("omni_reward_data_" + str(count), row['chosen']['value'], response_1, response_2, chosen, rejected, ground_truth, golden_rubric)

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
                        {"role": "user", "content": [
                            {"type": "image", "image": image_1,"min_pixels": 512 * 512, "max_pixels": 512 * 512,}, 
                            {"type": "text", "text": prompt}]
                        }
                    ],
                    "images": [image_1]
                }
            else:
                prompt = direct_judge_prompt.format(query=row['conversations'][-1]['value'],response_1=row['rejected']['value'],response_2=row['chosen']['value'])
                data = {
                    "conversations": [
                        {"role": "user", "content": [
                            {"type": "image", "image": image_1,"min_pixels": 512 * 512, "max_pixels": 512 * 512,}, 
                            {"type": "text", "text": prompt}]
                        }
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
    prepare_ti2t_sft(sft_data)
    # prepare_t2v_sft(sft_data)
    
    print(len(rl_data))
    random.shuffle(rl_data)
    rl_data = rl_data[:10000]
    
    prepare_language_sft(rl_data)
    print(len(rl_data))
    random.shuffle(rl_data)
    table = pa.Table.from_pylist(rl_data)
    pq.write_table(table, "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/criteria_TTS/omni_rl_train_data_v2.parquet")
    
prepare_omni_sft()
