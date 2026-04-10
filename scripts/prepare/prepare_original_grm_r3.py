import pyarrow as pa
import pyarrow.parquet as pq
import random
import re

THINKING_PRMOPT="""You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.\n\n\n### Important constraints (must follow exactly):\n1. Work only from the provided {query}, {response_1} (Assistant A), and {response_2} (Assistant B). Do not introduce outside facts or assumptions.\n2. Output only the structured block described below and nothing else (no preamble, no postscript).\n3. Use exactly three evaluation criteria (no more, no fewer). The criteria must be distinct (non-overlapping) and focused on observable differences between the two responses.\n3. For each criterion, provide: A short name, A one-sentence explanation of what it measures and why it matters. Each of analyses must explicitly identify the response’s strengths AND weaknesses, especially the concrete defects relevant to that criterion.\n4. In each analysis: When pointing out a defect, explain why it is a defect with respect to the criterion (e.g., “fails to answer X”, “contradicts user intent”, “includes factual error”, “provides irrelevant content”). If a response lacks relevant content, clearly state: “Response A/B lacks evidence for X.” Every Judge analysis must include: (a) what works, (b) what fails, (c) concrete evidence, and (d) criterion-based impact.\n5. After the three criteria blocks, give a single final verdict line containing exactly [[A]] or [[B]] (choose the response that better meets the query overall).\n\n\n### Required output format (produce exactly this structure — replace placeholders with real content):\n<Criteria 1> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 1>\n<Criteria 2> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 2>\n<Criteria 3> Name. Explanation. <Judge A>xxx</Judge A>\n<Judge B>xxx</Judge B></Criteria 3>\nThe final verdict is [[A]] or [[B]]\n\n\n### Input:[User Question]:\n{query}\n\n[The Start of Assistant A's Answer]:\n{response_1}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]: \n{response_2}\n[The End of Assistant B's Answer]\nPlease output your analysis and final verdict:"""


def build_criteria_TTS_conversation(idx, query, chosen, rejected, ground_truth):

    prompt = THINKING_PRMOPT.format(query=query, response_1=chosen, response_2=rejected)
    
    data = {
        "data_source": "Original_GRM",
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
    
    return data


def _extract(text):

    input_part = re.search(r"(?<=## INPUT)([\s\S]*?)(?=## RESPONSE 1)", text)
    resp1_part = re.search(r"(?<=## RESPONSE 1)([\s\S]*?)(?=## RESPONSE 2)", text)
    resp2_part = re.search(r"(?<=## RESPONSE 2)([\s\S]*?)(?=### EVALUATION)", text)

    return input_part.group(1).strip(), resp1_part.group(1).strip(), resp2_part.group(1).strip()


def build_lang_data():

    parquet_file = pq.ParquetFile("/workspace/mnt/lxb_work/hf_dir/hf_dataset/R3_20k/data/train-00000-of-00001.parquet")
    
    lang_train_data = []
    ground_truth_0 = ground_truth_1 = 0
    for batch in parquet_file.iter_batches(batch_size=10000):

        df = batch.to_pandas()
        for idx, row in df.iterrows():
            try:
                query, response_1, response_2 = _extract(row['prompt'])
            except:
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

    print(ground_truth_0, ground_truth_1)
    random.shuffle(lang_train_data)
    table = pa.Table.from_pylist(lang_train_data)
    pq.write_table(table, "/workspace/mnt/lxb_work/MindMirror/GRM-omni-train-v1/data/original_grm_r3_20k_train_data.parquet")

def main():

    build_lang_data()

if __name__ == "__main__":
    main()