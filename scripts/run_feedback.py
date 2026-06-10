#!/usr/bin/env python
"""Verification Feedback Framework pipeline."""

import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgebench.common.llm_client import LLMClient
from bgebench.common.task_loader import load_tasks, filter_tasks
from bgebench.feedback.run_baseline import run_baseline
from bgebench.feedback.run_repair_loop import run_feedback_loop
from bgebench.feedback.analyze_results import compare_baseline_vs_treatment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feedback")


def main():
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = config.get("llm", {}).get("api_key", "").replace("${DEEPSEEK_API_KEY}", "")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set")
        sys.exit(1)

    proxy = config.get("proxy", {})
    proxy_url = proxy.get("http", "") if proxy else ""

    llm_cfg = config.get("llm", {})
    client = LLMClient(
        api_key=api_key,
        base_url=llm_cfg.get("base_url", "https://api.deepseek.com"),
        model=llm_cfg.get("model", "deepseek-chat"),
        proxy=proxy_url or None,
    )

    bm = config.get("benchmarks", {})
    tasks = load_tasks(
        humaneval_path=bm.get("humaneval_path"),
        mbpp_path=bm.get("mbpp_path"),
    )
    tasks = filter_tasks(tasks, max_tasks=config.get("generation", {}).get("max_tasks"))

    if not tasks:
        logger.error("No tasks loaded. Download benchmark datasets first.")
        sys.exit(1)

    gen_cfg = config.get("generation", {})
    feedback_cfg = config.get("feedback", {})
    output_dir = Path(config.get("output", {}).get("data_dir", "data"))

    logger.info("Step 1/3: Running baseline (direct generation)...")
    baseline = run_baseline(
        tasks=tasks,
        llm_client=client,
        samples_per_task=gen_cfg.get("samples_per_task", 10),
        output_dir=output_dir,
        model_name=llm_cfg.get("model", "deepseek-chat"),
    )

    logger.info("Step 2/3: Running repair loop (treatment condition)...")
    ver_cfg = config.get("verification", {})
    iterations = run_feedback_loop(
        baseline_generations=baseline,
        tasks=tasks,
        llm_client=client,
        max_repair_iterations=feedback_cfg.get("max_repair_iterations", 3),
        tools=ver_cfg.get("tools"),
        output_dir=output_dir,
    )

    logger.info("Step 3/3: Comparing baseline vs treatment...")
    results = compare_baseline_vs_treatment(
        baseline_generations=baseline,
        repair_iterations=iterations,
        output_dir=output_dir / "results",
    )

    logger.info(
        "Feedback pipeline complete. Pass rate: baseline=%.4f -> treatment=%.4f",
        results.get("baseline_pass_rate", 0),
        results.get("treatment_pass_rate", 0),
    )


if __name__ == "__main__":
    main()
