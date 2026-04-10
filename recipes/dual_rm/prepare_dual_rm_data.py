import pyarrow as pa
import pyarrow.parquet as pq
import random
import re

THINKING_PRMOPT="""Given a user query and two responses, produce a comprehensive and well-structured set (ten) of evaluation rubric that can be used to distinguish the relative quality of the two responses.\nProvide a list of candidate rubrics ordered by importance from highest to lowest. These rubrics should be sufficiently discriminative to accurately capture meaningful differences between the responses.\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output:"""


def build_criteria_TTS_conversation(idx, query, response_1, response_2, ground_truth):

    prompt = THINKING_PRMOPT.format(query=query, response_1=response_1, response_2=response_2)

    assert ground_truth == 0 or ground_truth == 1
    data = {
        "data_source": "R3",
        "agent_name": "criteria_TTS",
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
            "idx":  idx,
            "interaction_kwargs": {
                "name": "criteria_TTS_interaction",
                "mode": "meta_reward",
                "query": query,
                "rejected": response_1 if ground_truth == 1 else response_2,
                "chosen": response_1 if ground_truth == 0 else response_2,
                "response_1": response_1,
                "response_2": response_2, 
                "ground_truth": ground_truth
            },
        }
    }
    
    return data

def _extract(text):
    input_part = re.search(r"(?<=## INPUT)([\s\S]*?)(?=## RESPONSE 1)", text)
    resp1_part = re.search(r"(?<=## RESPONSE 1)([\s\S]*?)(?=## RESPONSE 2)", text)
    resp2_part = re.search(r"(?<=## RESPONSE 2)([\s\S]*?)(?=### EVALUATION)", text)

    return input_part.group(1).strip(), resp1_part.group(1).strip(), resp2_part.group(1).strip()


def build_lang_data():

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/R3_20k/data/train-00000-of-00001.parquet")
    
    lang_train_data = []
    tie = ground_truth_0 = ground_truth_1 = 0
    for batch in parquet_file.iter_batches(batch_size=10000):

        df = batch.to_pandas()
        for idx, row in df.iterrows():

            try:
                query, response_1, response_2 = _extract(row['prompt'])
                import pdb; pdb.set_trace()
            except:
                tie += 1
                continue
            
            if "Response 1" in row['actual_score']:
                ground_truth = 0
            elif "Response 2" in row['actual_score']:
                ground_truth = 1
            else:
                continue
            
            data = build_criteria_TTS_conversation(idx, query, response_1, response_2, ground_truth)
            lang_train_data.append(data)
            
            if ground_truth == 1:
                ground_truth_1 += 1
            else:
                ground_truth_0 += 1
    
    print(ground_truth_0, ground_truth_1, tie)
    random.shuffle(lang_train_data)
    table = pa.Table.from_pylist(lang_train_data)
    pq.write_table(table, "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/criteria_TTS/r3_meta_reward_train_data.parquet")

def main():

    build_lang_data()

if __name__ == "__main__":
    main()