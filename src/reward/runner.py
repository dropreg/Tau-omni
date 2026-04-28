import asyncio
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ms_agent import LLMAgent
from ms_agent.config import Config
from ms_agent.llm.utils import Message

from .io import append_jsonl, disable_proxy, load_existing_results, read_jsonl
from .multimodal import ContentBundle, build_user_message_content, normalize_content
from .progress import FennecProgressBar
from .prompting import JUDGE_USER_PROMPT, parse_verdict


logging.getLogger("ms_agent").setLevel(logging.ERROR)


@dataclass
class GenRMRunConfig:
    query_file: str
    output_file: str
    agent_config_path: str
    ports: list[int]
    progress_width: int = 40


def build_base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/v1"


def build_agent(agent_config_path: str, base_url: str) -> LLMAgent:
    config = Config.from_task(agent_config_path)
    config.generation_config.stream = False
    config.callbacks = []
    config.max_chat_round = 1
    config.llm.openai_base_url = base_url
    return LLMAgent(config=config)


async def _run_agent(agent: LLMAgent, messages: list[Message]):
    return await agent.run(messages=messages)


def extract_record(item: dict) -> tuple[ContentBundle, ContentBundle, ContentBundle, str]:
    query_bundle = normalize_content(item["conversations"][0]["content"])
    chosen_bundle = normalize_content(item["chosen"]["content"])
    rejected_bundle = normalize_content(item["rejected"]["content"])

    if random.random() < 0.5:
        return query_bundle, chosen_bundle, rejected_bundle, "A"
    return query_bundle, rejected_bundle, chosen_bundle, "B"


def judge_one_with_agent(agent: LLMAgent, item: dict, idx: int, base_url: str) -> dict:
    start_time = time.perf_counter()
    suffix = str(item["suffix"])
    query_bundle, response_a_bundle, response_b_bundle, gold = extract_record(item)

    try:
        prompt_text = JUDGE_USER_PROMPT.format(
            query=query_bundle.text,
            response_1=response_a_bundle.text,
            response_2=response_b_bundle.text,
        )
        user_content = build_user_message_content(
            prompt_text=prompt_text,
            query_bundle=query_bundle,
            response_a_bundle=response_a_bundle,
            response_b_bundle=response_b_bundle,
        )
        response = asyncio.run(
            _run_agent(
                agent,
                [
                    Message(role="system", content=agent.config.prompt.system),
                    Message(role="user", content=user_content),
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
        "query": query_bundle.text,
        "response_1": response_a_bundle.text,
        "response_2": response_b_bundle.text,
        "query_media_count": len(query_bundle.media_blocks),
        "response_1_media_count": len(response_a_bundle.media_blocks),
        "response_2_media_count": len(response_b_bundle.media_blocks),
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
    progress_bar: FennecProgressBar,
    agent_config_path: str,
) -> None:
    base_url = build_base_url(port)
    agent = build_agent(agent_config_path, base_url)

    while True:
        task = task_queue.get()
        if task is None:
            task_queue.task_done()
            break

        idx, item = task
        result = judge_one_with_agent(agent, item, idx, base_url)
        results[idx] = result
        with file_lock:
            append_jsonl(result, output_file)
        progress_bar.update(error=result["error"] is not None)
        task_queue.task_done()


def run_genrm(config: GenRMRunConfig) -> None:
    disable_proxy()
    run_start_time = time.perf_counter()
    dataset = read_jsonl(config.query_file)
    Path(config.output_file).parent.mkdir(parents=True, exist_ok=True)
    existing_results = load_existing_results(config.output_file)
    progress_bar = FennecProgressBar(total=len(dataset), width=config.progress_width)

    task_queue: queue.Queue = queue.Queue()
    results: list[dict | None] = [None] * len(dataset)
    file_lock = threading.Lock()

    workers = []
    for port in config.ports:
        thread = threading.Thread(
            target=worker_loop,
            args=(
                port,
                task_queue,
                results,
                config.output_file,
                file_lock,
                progress_bar,
                config.agent_config_path,
            ),
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
            progress_bar.update(skipped=True, error=cached.get("error") is not None)
            continue
        task_queue.put((idx, item))

    for _ in config.ports:
        task_queue.put(None)

    task_queue.join()
    for thread in workers:
        thread.join()

    final_results = [item for item in results if item is not None]
    wall_time = time.perf_counter() - run_start_time
    progress_bar.finish()
    print_metrics(final_results)
    print(f"wall_time_seconds: {wall_time:.2f}")
    print(f"saved: {config.output_file}")
