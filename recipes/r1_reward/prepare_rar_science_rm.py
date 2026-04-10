import pyarrow as pa
import pyarrow.parquet as pq
import random
import json

THINKING_PRMOPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n###Output Format\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""


def build_lang_data():

    ground_truth_0 = ground_truth_1  = 0
    idx = 0
    lang_train_data = []
    for line in open("/workspace/mnt/lxb_work/GRM/GRM-omni-data/grm_omni_dataset/rar_science/rm_think.jsonl").readlines():

        json_item = json.loads(line)
        idx += 1
        if "The final verdict is [[A]]." == json_item['conversations'][1]['content']:
            ground_truth = 0
            ground_truth_0 += 1
        elif "The final verdict is [[B]]." == json_item['conversations'][1]['content']:
            ground_truth = 1
            ground_truth_1 += 1

        data = {
            "data_source": "R1_Reward",
            "prompt": [
                {
                    "role": "user",
                    "content": json_item['conversations'][0]['content'],
                }
            ],
            'reward_model': {
                'ground_truth': ground_truth,
            },
            "extra_info": {
                "idx":  idx
            }
        }
        lang_train_data.append(data)
    
    print(ground_truth_0, ground_truth_1)
    random.shuffle(lang_train_data)
    table = pa.Table.from_pylist(lang_train_data)
    pq.write_table(table, "/workspace/mnt/lxb_work/GRM/Tau-omni/data/original_rar_science_grm_train_data.parquet")

def main():

    build_lang_data()

if __name__ == "__main__":
    main()