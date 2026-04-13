import random

import pyarrow as pa
import pyarrow.parquet as pq


INPUT_PATH = "data/modelscope/Skywork-Reward-Preference-80K-v0.2/data/train-00000-of-00001.parquet"
OUTPUT_PATH = "data/modelscope/Skywork-Reward-Preference-80K-v0.2/data/train_criteria_tts.parquet"

THINKING_PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.


### Important constraints (must follow exactly):
1. Work only from the provided {query}, {response_1} (Assistant A), and {response_2} (Assistant B). Do not introduce outside facts or assumptions.
2. Output only the structured block described below and nothing else (no preamble, no postscript).
3. Use exactly three evaluation criteria (no more, no fewer). The criteria must be distinct (non-overlapping) and focused on observable differences between the two responses.
4. For each criterion, provide: A short name, A one-sentence explanation of what it measures and why it matters. Each analysis must explicitly identify the response's strengths AND weaknesses, especially the concrete defects relevant to that criterion.
5. In each analysis: When pointing out a defect, explain why it is a defect with respect to the criterion. If a response lacks relevant content, clearly state it.
6. After the three criteria blocks, give a single final verdict line containing exactly [[A]] or [[B]].


### Required output format:
<Criteria 1> Name. Explanation. <Judge A>xxx</Judge A>
<Judge B>xxx</Judge B></Criteria 1>
<Criteria 2> Name. Explanation. <Judge A>xxx</Judge A>
<Judge B>xxx</Judge B></Criteria 2>
<Criteria 3> Name. Explanation. <Judge A>xxx</Judge A>
<Judge B>xxx</Judge B></Criteria 3>
The final verdict is [[A]] or [[B]]


### Input:[User Question]:
{query}

[The Start of Assistant A's Answer]:
{response_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]:
{response_2}
[The End of Assistant B's Answer]
Please output your analysis and final verdict:"""


def build_criteria_tts_conversation(idx, query, chosen, rejected):
    if random.random() > 0.5:
        ground_truth = 1
        prompt = THINKING_PROMPT.format(
            query=query,
            response_1=rejected,
            response_2=chosen,
        )
    else:
        ground_truth = 0
        prompt = THINKING_PROMPT.format(
            query=query,
            response_1=chosen,
            response_2=rejected,
        )

    data = {
        "data_source": "Skywork-Reward-Preference-80K-v0.2",
        "agent_name": "criteria_TTS",
        "prompt": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "reward_model": {
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "idx": idx,
            "interaction_kwargs": {
                "name": "grm_omni_criteria_TTS_intraction",
                "query": query,
                "rejected": rejected,
                "chosen": chosen,
                "ground_truth": ground_truth,
            },
        },
    }
    return data, ground_truth


def build_lang_data():
    parquet_file = pq.ParquetFile(INPUT_PATH)

    lang_train_data = []
    ground_truth_0 = 0
    ground_truth_1 = 0

    for batch in parquet_file.iter_batches(batch_size=10000):
        df = batch.to_pandas()

        for idx, row in df.iterrows():
            if len(row["rejected"]) != 2:
                continue

            query = row["chosen"][0]["content"]
            chosen = row["chosen"][1]["content"]
            rejected = row["rejected"][1]["content"]

            data, ground_truth = build_criteria_tts_conversation(
                idx=idx,
                query=query,
                chosen=chosen,
                rejected=rejected,
            )
            lang_train_data.append(data)

            if ground_truth == 1:
                ground_truth_1 += 1
            else:
                ground_truth_0 += 1

    print(f"ground_truth=0: {ground_truth_0}")
    print(f"ground_truth=1: {ground_truth_1}")
    print(f"total: {len(lang_train_data)}")

    random.shuffle(lang_train_data)
    table = pa.Table.from_pylist(lang_train_data)
    pq.write_table(table, OUTPUT_PATH)
    print(f"Saved to: {OUTPUT_PATH}")


def main():
    build_lang_data()


if __name__ == "__main__":
    main()
