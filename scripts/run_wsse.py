#!/usr/bin/env python
"""WSSE 2026: Bug Growth Elasticity measurement pipeline."""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgebench.common.llm_client import LLMClient
from bgebench.common.task_loader import load_tasks, filter_tasks
from bgebench.wsse.run_generation import generate_samples
from bgebench.wsse.run_verification import verify_generations
from bgebench.wsse.analyze_growth import analyze_growth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wsse")


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
    output_dir = Path(config.get("output", {}).get("data_dir", "data"))

    logger.info("Step 1/3: Generating code samples...")
    generations = generate_samples(
        tasks=tasks,
        llm_client=client,
        samples_per_task=gen_cfg.get("samples_per_task", 10),
        prompt_variants=gen_cfg.get("prompt_variants", ["direct"]),
        output_dir=output_dir,
        model_name=llm_cfg.get("model", "deepseek-chat"),
    )

    logger.info("Step 2/3: Running verification tools...")
    ver_cfg = config.get("verification", {})
    all_tool_results, all_defects = verify_generations(
        generations=generations,
        tasks=tasks,
        tools=ver_cfg.get("tools"),
        output_dir=output_dir,
    )

    logger.info("Step 3/3: Analyzing BGE growth models...")
    generations_df = pd.read_csv(output_dir / "results" / "generations.csv")
    defects_df = pd.read_csv(output_dir / "results" / "defects.csv")

    results = analyze_growth(
        generations_df=generations_df,
        defects_df=defects_df,
        output_dir=output_dir / "results",
    )

    best_bge = results.get("power_law", {}).get("bge", "N/A")
    logger.info("WSSE pipeline complete. BGE = %s", best_bge)


if __name__ == "__main__":
    main()
