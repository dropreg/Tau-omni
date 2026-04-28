import asyncio
import json
import logging
import os
import queue
import random
import re
import threading
import time
from pathlib import Path

from ms_agent import LLMAgent
from ms_agent.config import Config
from ms_agent.llm.utils import Message


logging.getLogger("ms_agent").setLevel(logging.ERROR)


QUERY_FILE = "data/benchmark/rewardbench/chat.jsonl"
OUTPUT_FILE = "output/gen_rm/rewardbench_qwen3_results.jsonl"
PORTS = [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
AGENT_CONFIG = "recipes/gen_rm/agent.yaml"

JUDGE_USER_PROMPT = """请比较下面同一用户问题的两个候选回答，并判断哪个回答整体更好。

[User Query]
{query}

[Assistant A]
{response_1}

[Assistant B]
{response_2}

请先给出简要分析，最后单独输出一行：
Winner: [[A]] / [[B]] / [[Tie]]"""

VERDICT_PATTERN = re.compile(r"\[\[\s*(A|B|Tie)\s*\]\]", re.IGNORECASE)


def disable_proxy() -> None:
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"


def build_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def read_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_existing_results(path: str) -> dict[str, dict]:
    existing = {}
    output_path = Path(path)
    if not output_path.exists():
        return existing

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            suffix = item.get("suffix")
            if suffix is not None:
                existing[str(suffix)] = item
    return existing


def extract_record(item: dict) -> tuple[str, str, str, str]:
    query = item["conversations"][0]["content"]
    chosen = item["chosen"]["content"]
    rejected = item["rejected"]["content"]

    if random.random() < 0.5:
        return query, chosen, rejected, "A"
    return query, rejected, chosen, "B"


def parse_verdict(text: str) -> str | None:
    matches = VERDICT_PATTERN.findall(text)
    if not matches:
        return None
    verdict = matches[-1].strip().lower()
    if verdict == "tie":
        return "Tie"
    if verdict == "a":
        return "A"
    if verdict == "b":
        return "B"
    return None


def build_agent(base_url: str) -> LLMAgent:
    config = Config.from_task(AGENT_CONFIG)
    config.generation_config.stream = False
    config.callbacks = []
    config.max_chat_round = 1
    config.llm.openai_base_url = base_url
    return LLMAgent(config=config)


async def _run_agent(agent: LLMAgent, messages: list[Message]):
    return await agent.run(messages=messages)


def judge_one_with_agent(agent: LLMAgent, item: dict, idx: int, base_url: str) -> dict:
    start_time = time.perf_counter()
    suffix = str(item["suffix"])
    query, response_1, response_2, gold = extract_record(item)

    try:
        user_prompt = JUDGE_USER_PROMPT.format(
            query=query,
            response_1=response_1,
            response_2=response_2,
        )
        response = asyncio.run(
            _run_agent(
                agent,
                [
                    Message(role="system", content=agent.config.prompt.system),
                    Message(role="user", content=user_prompt),
                ],
            )
        )
        last = response[-1]
        reply = last.content
        pred = parse_verdict(reply)
        prompt_tokens = getattr(last, "prompt_tokens", None)
        completion_tokens = getattr(last, "completion_tokens", None)
        error = None
    except Exception as exc:
        reply = ""
        pred = None
        prompt_tokens = None
        completion_tokens = None
        error = repr(exc)

    elapsed_seconds = time.perf_counter() - start_time
    correct = pred == gold

    return {
        "idx": idx,
        "suffix": suffix,
        "query": query,
        "response_1": response_1,
        "response_2": response_2,
        "gold": gold,
        "pred": pred,
        "correct": correct,
        "base_url": base_url,
        "reply": reply,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "error": error,
        "elapsed_seconds": elapsed_seconds,
    }


def append_result(result: dict, output_file: str, file_lock: threading.Lock) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def print_metrics(results: list[dict]) -> None:
    total = len(results)
    parsed = sum(item["pred"] is not None for item in results)
    correct = sum(bool(item["correct"]) for item in results)
    tie_count = sum(item["pred"] == "Tie" for item in results)
    a_count = sum(item["pred"] == "A" for item in results)
    b_count = sum(item["pred"] == "B" for item in results)
    error_count = sum(item["error"] is not None for item in results)
    total_elapsed = sum(float(item.get("elapsed_seconds", 0.0)) for item in results)
    avg_elapsed = total_elapsed / total if total else 0.0

    acc = correct / total if total else 0.0
    parsed_rate = parsed / total if total else 0.0

    print(f"total: {total}")
    print(f"parsed: {parsed}")
    print(f"parsed_rate: {parsed_rate:.4f}")
    print(f"pred_A: {a_count}")
    print(f"pred_B: {b_count}")
    print(f"pred_Tie: {tie_count}")
    print(f"errors: {error_count}")
    print(f"acc: {acc:.4f}")
    print(f"sum_elapsed_seconds: {total_elapsed:.2f}")
    print(f"avg_elapsed_seconds: {avg_elapsed:.2f}")


def worker_loop(
    port: int,
    task_queue: queue.Queue,
    results: list[dict | None],
    output_file: str,
    file_lock: threading.Lock,
    total: int,
) -> None:
    base_url = build_base_url(port)
    agent = build_agent(base_url)

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        idx, item = task
        print(f"[Running {idx + 1}/{total}] url={base_url}")
        result = judge_one_with_agent(agent, item, idx, base_url)
        results[idx] = result
        append_result(result, output_file, file_lock)
        print(
            f"[Done {idx + 1}/{total}] "
            f"gold={result['gold']} pred={result['pred']} "
            f"correct={result['correct']} error={result['error'] is not None}"
        )
        task_queue.task_done()


def run_all() -> None:
    run_start_time = time.perf_counter()
    dataset = read_jsonl(QUERY_FILE)
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_results = load_existing_results(OUTPUT_FILE)

    task_queue: queue.Queue = queue.Queue()
    results: list[dict | None] = [None] * len(dataset)
    file_lock = threading.Lock()

    workers = []
    for port in PORTS:
        thread = threading.Thread(
            target=worker_loop,
            args=(port, task_queue, results, OUTPUT_FILE, file_lock, len(dataset)),
            daemon=True,
        )
        thread.start()
        workers.append(thread)

    for idx, item in enumerate(dataset):
        suffix = str(item["suffix"])
        if suffix in existing_results:
            cached = existing_results[suffix]
            cached["idx"] = idx
            cached["suffix"] = suffix
            results[idx] = cached
            print(f"[Skip {idx + 1}/{len(dataset)}] suffix={suffix}")
            continue
        task_queue.put((idx, item))

    for _ in PORTS:
        task_queue.put(None)

    task_queue.join()

    for thread in workers:
        thread.join()

    final_results = [item for item in results if item is not None]
    wall_time = time.perf_counter() - run_start_time
    print_metrics(final_results)
    print(f"wall_time_seconds: {wall_time:.2f}")
    print(f"saved: {OUTPUT_FILE}")


def main() -> None:
    disable_proxy()
    run_all()


if __name__ == "__main__":
    main()
