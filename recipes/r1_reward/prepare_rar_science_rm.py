import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = "data/r1_reward/RaR-Science-Preference.jsonl"
DEFAULT_OUTPUT_PATH = "data/r1_reward/RaR-Science-Preference.parquet"
DEFAULT_DATA_SOURCE = "RaR-Science"

LABEL_TO_VERDICT = {0: "A", 1: "B"}
VERDICT_TO_LABEL = {"A": 0, "B": 1}

JUDGE_PROMPT = """You are an objective, impartial, and unbiased content evaluator. Given a user query and two candidate assistant responses, produce a rigorous, evidence-based comparison and a single final verdict indicating which response better fulfills the user's intent.

### Output Format
The final verdict is [[A]] or [[B]]

### Input: [User Question]:
{query}

[The Start of Assistant A's Answer]:
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]:
{response_b}
[The End of Assistant B's Answer]

Please output your analysis and final verdict:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RaR-Science preference data into verl-compatible parquet "
            "for R1-Reward / RM-R1 training."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="Input jsonl path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output parquet path.")
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help="Value written to each record's data_source field.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for pair order shuffling and final record shuffle.",
    )
    parser.add_argument(
        "--pair-order",
        choices=("preserve", "shuffle"),
        default="preserve",
        help=(
            "How to order chosen/rejected pair records when the input is not already "
            "a conversations-style judge prompt."
        ),
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip malformed lines instead of raising an error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report statistics without writing parquet.",
    )
    return parser.parse_args()


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text_content", "value", "text"):
            if key in value:
                return as_text(value[key])
    if isinstance(value, list):
        return "\n\n".join(as_text(item) for item in value)
    if value is None:
        return ""
    return str(value)


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def extract_verdict(text: str) -> str | None:
    matches = re.findall(r"\[{1,2}([aAbB])\]{1,2}", text)
    if matches:
        return matches[-1].upper()

    normalized = text.strip().upper()
    if normalized in VERDICT_TO_LABEL:
        return normalized
    return None


def normalize_label(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in LABEL_TO_VERDICT:
        return value
    if isinstance(value, float) and int(value) in LABEL_TO_VERDICT:
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value in ("0", "1"):
            return int(value)
        verdict = extract_verdict(value)
        if verdict is not None:
            return VERDICT_TO_LABEL[verdict]
    return None


def build_verl_record(
    *,
    idx: int,
    data_source: str,
    prompt: str,
    ground_truth: int,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if ground_truth not in LABEL_TO_VERDICT:
        raise ValueError(f"Line {idx} has unsupported ground_truth: {ground_truth!r}")

    return {
        "data_source": data_source,
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
            **(extra_info or {}),
        },
    }


def from_conversations(
    item: dict[str, Any],
    *,
    idx: int,
    data_source: str,
) -> dict[str, Any]:
    conversations = item["conversations"]
    if len(conversations) < 2:
        raise ValueError(f"Line {idx} has fewer than 2 conversation turns.")

    prompt = as_text(conversations[0].get("content", ""))
    verdict = extract_verdict(as_text(conversations[1].get("content", "")))
    if verdict is None:
        raise ValueError(f"Line {idx} has no final verdict. Expected [[A]] or [[B]].")

    return build_verl_record(
        idx=idx,
        data_source=data_source,
        prompt=prompt,
        ground_truth=VERDICT_TO_LABEL[verdict],
        extra_info={
            "format": "conversations",
            "verdict": verdict,
        },
    )


def from_pair_record(
    item: dict[str, Any],
    *,
    idx: int,
    data_source: str,
    pair_order: str,
) -> dict[str, Any]:
    paired_data = item.get("paired_data", {})
    source = paired_data if isinstance(paired_data, dict) else {}

    query = first_present(item, ("query", "question", "prompt", "instruction"))
    query = query if query is not None else first_present(source, ("query", "question", "prompt", "instruction"))

    response_a = first_present(item, ("response_1", "response1", "answer_a", "answer_A"))
    response_b = first_present(item, ("response_2", "response2", "answer_b", "answer_B"))
    response_a = response_a if response_a is not None else first_present(source, ("response_1", "response1", "answer_a", "answer_A"))
    response_b = response_b if response_b is not None else first_present(source, ("response_2", "response2", "answer_b", "answer_B"))

    label = normalize_label(
        first_present(item, ("ground_truth", "label", "winner", "verdict", "preferred"))
    )
    if label is None:
        label = normalize_label(
            first_present(source, ("ground_truth", "label", "winner", "verdict", "preferred"))
        )

    chosen = first_present(item, ("chosen", "better", "positive"))
    rejected = first_present(item, ("rejected", "worse", "negative"))
    chosen = chosen if chosen is not None else first_present(source, ("chosen", "better", "positive"))
    rejected = rejected if rejected is not None else first_present(source, ("rejected", "worse", "negative"))

    if response_a is None or response_b is None:
        if chosen is None or rejected is None:
            raise ValueError(
                f"Line {idx} must provide response_1/response_2 or chosen/rejected."
            )

        if pair_order == "shuffle" and random.random() < 0.5:
            response_a, response_b = rejected, chosen
            label = 1
        else:
            response_a, response_b = chosen, rejected
            label = 0

    if label is None:
        raise ValueError(
            f"Line {idx} must provide ground_truth/label/winner for response_1/response_2 pairs."
        )

    prompt = JUDGE_PROMPT.format(
        query=as_text(query),
        response_a=as_text(response_a),
        response_b=as_text(response_b),
    )

    return build_verl_record(
        idx=idx,
        data_source=data_source,
        prompt=prompt,
        ground_truth=label,
        extra_info={
            "format": "pair",
            "verdict": LABEL_TO_VERDICT[label],
        },
    )


def load_record(
    line: str,
    *,
    idx: int,
    data_source: str,
    pair_order: str,
) -> dict[str, Any]:
    item = json.loads(line)
    if "conversations" in item:
        return from_conversations(item, idx=idx, data_source=data_source)
    return from_pair_record(
        item,
        idx=idx,
        data_source=data_source,
        pair_order=pair_order,
    )


def build_parquet(
    *,
    input_path: Path,
    output_path: Path,
    data_source: str,
    seed: int,
    pair_order: str,
    skip_invalid: bool,
    dry_run: bool,
) -> None:
    random.seed(seed)
    label_counts = {0: 0, 1: 0}
    records = []
    skipped = 0

    with input_path.open("r", encoding="utf-8") as reader:
        for idx, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = load_record(
                    line,
                    idx=idx,
                    data_source=data_source,
                    pair_order=pair_order,
                )
            except Exception:
                if not skip_invalid:
                    raise
                skipped += 1
                continue

            label_counts[record["reward_model"]["ground_truth"]] += 1
            records.append(record)

    random.shuffle(records)
    if not dry_run:
        write_parquet(records, output_path)
        print(f"Saved {len(records)} records to {output_path}")
    else:
        print(f"Parsed {len(records)} records from {input_path}")

    print(f"Label counts: A={label_counts[0]}, B={label_counts[1]}")
    if skipped:
        print(f"Skipped invalid records: {skipped}")


def write_parquet(records: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to write parquet. Install it with "
            "`pip install pyarrow` or `pip install -r requirement`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), output_path)


def main() -> None:
    args = parse_args()
    build_parquet(
        input_path=Path(args.input),
        output_path=Path(args.output),
        data_source=args.data_source,
        seed=args.seed,
        pair_order=args.pair_order,
        skip_invalid=args.skip_invalid,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
