from src.reward import GenRMRunConfig, run_genrm


def main() -> None:
    config = GenRMRunConfig(
        query_file="data/benchmark/rewardbench/chat.jsonl",
        output_file="output/gen_rm/rewardbench_qwen3_results.jsonl",
        agent_config_path="recipes/gen_rm/inference/agent.yaml",
        ports=[8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007],
        progress_width=40,
    )
    run_genrm(config)


if __name__ == "__main__":
    main()
