import json
import random
import pyarrow.parquet as pq
import random
import re


PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n###Output Format\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:\n\n[Context]:\n{context}\n\n[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""


def build_math_step_dpo_data(train_data):

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/Math-Step-DPO-10K/data/train-00000-of-00001.parquet")
    count = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        
        df = batch.to_pandas()
        for idx, row in df.iterrows():
            
            chosen = row['initial_reason_steps'] + row['full_chosen']
            rejected = row['initial_reason_steps'] + row['full_rejected']
            query = row['prompt']

            if random.random() > 0.5:
                response_1 = chosen
                response_2 = rejected 
                verdict = "The final verdict is [[A]]."
            else:
                response_1 = rejected
                response_2 = chosen
                verdict = "The final verdict is [[B]]."

            prompt = PROMPT.format(context="", query=query, response_1=response_1, response_2=response_2)
            data = {"conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": verdict}
            ]}
            train_data.append(data)
            count += 1
    print(count)


def _extract(text):
    input_part = re.search(r"(?<=## INPUT)([\s\S]*?)(?=## RESPONSE 1)", text)
    resp1_part = re.search(r"(?<=## RESPONSE 1)([\s\S]*?)(?=## RESPONSE 2)", text)
    resp2_part = re.search(r"(?<=## RESPONSE 2)([\s\S]*?)(?=### EVALUATION)", text)

    return input_part.group(1).replace("\n\n#", "").strip(), resp1_part.group(1).replace("\n\n#", "").strip(), resp2_part.group(1).replace("\n\n#", "").strip()

def build_r3_data(train_data):

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/R3_20k/data/train-00000-of-00001.parquet")
    count = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        
        df = batch.to_pandas()
        for idx, row in df.iterrows():

            try:
                query, response_1, response_2 = _extract(row['prompt'])
            except:
                continue

            if "Response 1" in row['actual_score']:
                verdict = "The final verdict is [[A]]."
            elif "Response 2" in row['actual_score']:
                verdict = "The final verdict is [[B]]."
            else:
                continue

            prompt = PROMPT.format(context="", query=query, response_1=response_1, response_2=response_2)
            data = {"conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": verdict}
            ]}
            train_data.append(data)
            count += 1
    print(count)


def build_skywork_data(train_data):

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/Skywork-Reward-Preference-80K-v0.2/data/train-00000-of-00001.parquet")
    count = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        
        df = batch.to_pandas()
        for idx, row in df.iterrows():

            if len(row['chosen']) == 2:
                if random.random() > 0.5:
                    response_1 = row['chosen'][1]['content']
                    response_2 = row['rejected'][1]['content']
                    verdict = "The final verdict is [[A]]."
                else:
                    response_1 = row['rejected'][1]['content']
                    response_2 = row['chosen'][1]['content']
                    verdict = "The final verdict is [[B]]."

                prompt = PROMPT.format(context="", query=row['chosen'][0]['content'], response_1=response_1, response_2=response_2)
                data = {"conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": verdict}
                ]}
                train_data.append(data)
                count += 1
    print(count)

def build_UltraInteract_data(train_data):

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/UltraInteract_pair/0000_pair.parquet")
    count = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        
        df = batch.to_pandas()
        for idx, row in df.iterrows():
 
            query = row["trajectory"][0]['value']

            if random.random() > 0.5:
                response_1 = row['chosen']
                response_2 = row['rejected']
                verdict = "The final verdict is [[A]]."
            else:
                response_1 = row['rejected']
                response_2 = row['chosen']
                verdict = "The final verdict is [[B]]."

            prompt = PROMPT.format(context="", query=query, response_1=response_1, response_2=response_2)
            data = {"conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": verdict}
            ]}
            count += 1
            train_data.append(data)
    print(count)


def build_helpsteer(train_data):

    count = 0
    for line in open("/workspace/mnt/lxb_work/hf_dir/hf_dataset/HelpSteer3/preference/train.jsonl").readlines():
        
        json_item = json.loads(line)
        response_1 = json_item['response1']
        response_2 = json_item['response2']
        if len(json_item['context']) == 1:
            context = []
        else:
            context = "\n".join([c['content'] for c in json_item['context'][:-1]])
        query = json_item['context'][-1]['content']

        if json_item['overall_preference'] < 0:
            verdict = "The final verdict is [[A]]."
        else:
            verdict = "The final verdict is [[B]]."

        prompt = PROMPT.format(context=context, query=query, response_1=response_1, response_2=response_2)
        data = {"conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": verdict}
        ]}
        count += 1
        train_data.append(data)
    print(count)


def build_offsetbias(train_data):

    count = 0
    for line in open("/workspace/mnt/lxb_work/hf_dir/hf_dataset/offsetbias/train.jsonl").readlines():
        
        json_item = json.loads(line)
        response_1 = json_item['output_1']
        response_2 = json_item['output_2']
        query = json_item['instruction']

        if json_item['label'] == 1:
            verdict = "The final verdict is [[A]]."
        else:
            verdict = "The final verdict is [[B]]."

        prompt = PROMPT.format(context="", query=query, response_1=response_1, response_2=response_2)
        data = {"conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": verdict}
        ]}
        count += 1
        train_data.append(data)
    print(count)

train_data = []
build_offsetbias(train_data)
build_helpsteer(train_data)
build_math_step_dpo_data(train_data)
build_r3_data(train_data)
build_skywork_data(train_data)
build_UltraInteract_data(train_data)

print(len(train_data))
random.shuffle(train_data)
with open("/workspace/mnt/lxb_work/Tau-omni/data/all/sft.jsonl", 'w') as f:
    for data in train_data:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

