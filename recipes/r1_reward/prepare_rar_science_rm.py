import argparse
import json
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_PATH = (
    "/workspace/mnt/lxb_work/GRM/GRM-omni-data/"
    "grm_omni_dataset/rar_science/rm_think.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    "/workspace/mnt/lxb_work/GRM/Tau-omni/data/"
    "original_rar_science_grm_train_data.parquet"
)

DATA_SOURCE = "R1_Reward"
VERDICT_TO_LABEL = {"A": 0, "B": 1}

THINKING_PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user's intent.


### Output Format
The final verdict is [[A]] or [[B]]


### Input: [User Question]:
{query}

[The Start of Assistant A's Answer]:
{response_1}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]:
{response_2}
[The End of Assistant B's Answer]
Please output your analysis and final verdict:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAR-Science RM jsonl data into verl-compatible parquet."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help="Input jsonl path. Each line should contain a conversations field.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used before shuffling records.",
    )
    return parser.parse_args()


def load_record(line: str, idx: int) -> dict:
    item = json.loads(line)
    conversations = item["conversations"]

    if len(conversations) < 2:
        raise ValueError(f"Line {idx} has fewer than 2 conversation turns.")

    verdict = extract_verdict(conversations[1]["content"])
    if verdict is None:
        raise ValueError(
            f"Line {idx} has no supported verdict. Expected [[A]] or [[B]]."
        )

    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": conversations[0]["content"],
            }
        ],
        "reward_model": {
            "ground_truth": VERDICT_TO_LABEL[verdict],
        },
        "extra_info": {
            "idx": idx,
        },
    }


def extract_verdict(text: str) -> str | None:
    matches = re.findall(r"\[{1,2}([aAbB])\]{1,2}", text)
    if not matches:
        return None
    return matches[-1].upper()


def build_lang_data(input_path: Path, output_path: Path, seed: int) -> None:
    label_counts = {0: 0, 1: 0}
    records = []

    with input_path.open("r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue

            record = load_record(line, idx)
            label_counts[record["reward_model"]["ground_truth"]] += 1
            records.append(record)

    random.seed(seed)
    random.shuffle(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, output_path)

    print(f"Saved {len(records)} records to {output_path}")
    print(f"Label counts: A={label_counts[0]}, B={label_counts[1]}")


def main() -> None:
    args = parse_args()
    build_lang_data(
        input_path=Path(args.input),
        output_path=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
