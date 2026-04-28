import json
import random
from pathlib import Path

import pyarrow.parquet as pq


SKYWORK_PATH = (
    "/workspace/mnt/lxb_work/hf_dir/hf_dataset/"
    "Skywork-Reward-Preference-80K-v0.2/data/train-00000-of-00001.parquet"
)
OFFSETBIAS_PATH = "/workspace/mnt/lxb_work/hf_dir/hf_dataset/offsetbias/train.jsonl"
OUTPUT_PATH = "/workspace/mnt/lxb_work/Tau-omni/data/genrm/genrm_sft_skywork.jsonl"


PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user’s intent.


### Output Format
The final verdict is [[A]] or [[B]]


### Input:

[Context]:
{context}

[User Question]:
{query}

[The Start of Assistant A's Answer]:
{response_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]:
{response_2}
[The End of Assistant B's Answer]
Please output your analysis and final verdict:"""


def build_prompt(query: str, response_1: str, response_2: str, context: str = "") -> str:
    return PROMPT.format(
        context=context,
        query=query,
        response_1=response_1,
        response_2=response_2,
    )


def build_sample(prompt: str, verdict: str) -> dict:
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": verdict},
        ]
    }


def choose_order(chosen: str, rejected: str) -> tuple[str, str, str]:
    if random.random() > 0.5:
        return chosen, rejected, "The final verdict is [[A]]."
    return rejected, chosen, "The final verdict is [[B]]."


def load_skywork_data() -> list[dict]:
    dataset = []
    parquet_file = pq.ParquetFile(SKYWORK_PATH)

    for batch in parquet_file.iter_batches(batch_size=10000):
        df = batch.to_pandas()
        for _, row in df.iterrows():
            if len(row["chosen"]) != 2 or len(row["rejected"]) != 2:
                continue

            query = row["chosen"][0]["content"]
            chosen = row["chosen"][1]["content"]
            rejected = row["rejected"][1]["content"]
            response_1, response_2, verdict = choose_order(chosen, rejected)

            prompt = build_prompt(
                query=query,
                response_1=response_1,
                response_2=response_2,
            )
            dataset.append(build_sample(prompt, verdict))

    print(f"skywork: {len(dataset)}")
    return dataset


def load_offsetbias_data() -> list[dict]:
    dataset = []

    with open(OFFSETBIAS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            query = item["instruction"]
            response_1 = item["output_1"]
            response_2 = item["output_2"]
            verdict = "The final verdict is [[A]]." if item["label"] == 1 else "The final verdict is [[B]]."

            prompt = build_prompt(
                query=query,
                response_1=response_1,
                response_2=response_2,
            )
            dataset.append(build_sample(prompt, verdict))

    print(f"offsetbias: {len(dataset)}")
    return dataset


def save_jsonl(data: list[dict], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    train_data = []
    train_data.extend(load_offsetbias_data())
    train_data.extend(load_skywork_data())

    random.shuffle(train_data)
    save_jsonl(train_data, OUTPUT_PATH)

    print(f"total: {len(train_data)}")
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
