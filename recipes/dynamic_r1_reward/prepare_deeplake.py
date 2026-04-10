import pyarrow as pa
import pyarrow.parquet as pq
import random
import deeplake
import numpy as np
import os
import json



THINKING_PRMOPT="""You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Important constraints (must follow exactly):\n1. Work only from the provided, (Assistant A), and (Assistant B). Do not introduce outside facts or assumptions.\n2. Output only the structured block described below and nothing else (no preamble, no postscript).\n3. Use exactly three evaluation criteria (no more, no fewer). The criteria must be distinct (non-overlapping) and focused on observable differences between the two responses.\n3. For each criterion, provide: A short name, A one-sentence explanation of what it measures and why it matters. Each of analyses must explicitly identify the response’s strengths AND weaknesses, especially the concrete defects relevant to that criterion.\n4. In each analysis: When pointing out a defect, explain why it is a defect with respect to the criterion (e.g., “fails to answer X”, “contradicts user intent”, “includes factual error”, “provides irrelevant content”). If a response lacks relevant content, clearly state: “Response A/B lacks evidence for X.” Every Judge analysis must include: (a) what works, (b) what fails, (c) concrete evidence, and (d) criterion-based impact.\n5. After the three criteria blocks, give a single final verdict line containing exactly [[A]] or [[B]] (choose the response that better meets the query overall).\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Criteria 1> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 1>\n<Criteria 2> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 2>\n<Criteria 3> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 3>\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""


def build_criteria_TTS_conversation(idx, query, chosen, rejected):

    if random.random() > 0.5:
        ground_truth = 1
        prompt = THINKING_PRMOPT.format(query=query, response_1=rejected, response_2=chosen)
    else:
        ground_truth = 0
        prompt = THINKING_PRMOPT.format(query=query, response_1=chosen, response_2=rejected)
    
    data = {
        "data_source": "R1_Reward",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        'reward_model': {
            'ground_truth': ground_truth,
        },
        "extra_info": {
            "idx":  idx
        }
    }
    
    return data, ground_truth

def build_lang_data():
    
    try:
        ds = deeplake.create('/workspace/mnt/lxb_work/Tau-omni/examples/dynamic_r1_reward/skywork_deeplake')
    except:
        ds = deeplake.open('/workspace/mnt/lxb_work/Tau-omni/examples/dynamic_r1_reward/skywork_deeplake')
    
    ds.add_column('idx', dtype='text')
    ds.add_column('json_data', dtype='text')
    
    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/Skywork-Reward-Preference-80K-v0.2/data/train-00000-of-00001.parquet")

    lang_train_data = []
    ground_truth_0 = ground_truth_1 = 0

    for batch in parquet_file.iter_batches(batch_size=10000):
        
        df = batch.to_pandas()
        for idx, row in df.iterrows():
            if len(row['rejected']) == 2:
                
                data, ground_truth = build_criteria_TTS_conversation(idx, row['chosen'][0]['content'], row['chosen'][1]['content'], row['rejected'][1]['content'])
                lang_train_data.append(data)
                ds.append({
                    'idx': [str(idx)],
                    'json_data': [json.dumps(data)]
                })

                if ground_truth == 1:
                    ground_truth_1 += 1
                else:
                    ground_truth_0 += 1

    print(ground_truth_0, ground_truth_1)
    print(f"数据集构建完成，当前条数: {len(ds)}")

def update_lange_data():

    ds_path = '/workspace/mnt/lxb_work/Tau-omni/examples/dynamic_r1_reward/skywork_deeplake'
    ds = deeplake.open(ds_path)
    num_samples = len(ds)

    for i in range(num_samples):
        
        json_str = ds['json_data'][i]
        data = json.loads(json_str)
        # import pdb; pdb.set_trace()
        data["data_source"] = "Dynamic-Reward"
        ds['json_data'][i] = json.dumps(data)
        
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} / {num_samples} 条数据...")
    
    ds.commit("Update data_source to Dynamic")
    print(f"更新完成！共处理 {num_samples} 条。")

def main():

    # build_lang_data()
    update_lange_data()

if __name__ == "__main__":
    main()