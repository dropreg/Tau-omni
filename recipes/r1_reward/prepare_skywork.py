import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "Skywork/Skywork-Reward-Preference-80K-v0.1"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = "data/r1_reward/skywork/train.jsonl"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Skywork reward preference data via Hugging Face mirror "
            "and export it as train.jsonl."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output jsonl path.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help="Mirror endpoint used for Hugging Face Hub access.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for exported samples.",
    )
    return parser.parse_args()


def ensure_python_types(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ensure_python_types(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [ensure_python_types(item) for item in value]

    # datasets / numpy scalar compatibility
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value


def load_split(dataset_name: str, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets is required for downloading from Hugging Face. "
            "Install it with `pip install datasets`."
        ) from exc

    return load_dataset(dataset_name, split=split, cache_dir=cache_dir)


def export_jsonl(dataset, output_path: Path, max_samples: int | None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for row in dataset:
            record = ensure_python_types(dict(row))
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if max_samples is not None and count >= max_samples:
                break

    return count


def main() -> None:
    args = parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    dataset = load_split(
        dataset_name=args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    output_path = Path(args.output)
    count = export_jsonl(
        dataset=dataset,
        output_path=output_path,
        max_samples=args.max_samples,
    )

    print(f"HF_ENDPOINT={args.hf_endpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Saved {count} samples to {output_path}")


if __name__ == "__main__":
    main()
