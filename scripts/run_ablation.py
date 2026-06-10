#!/usr/bin/env python
"""Ablation experiment: compare feedback loop under different tool subsets."""

import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgebench.common.llm_client import LLMClient
from bgebench.common.task_loader import load_tasks, filter_tasks
from bgebench.common.schemas import Condition, Generation
from bgebench.common.task_loader import build_generation_prompt
from bgebench.common.storage import save_generations_csv, save_generated_code
from bgebench.feedback.run_repair_loop import run_feedback_loop
from bgebench.feedback.analyze_results import compare_baseline_vs_treatment

import csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ablation")


ABLATION_CONFIGS = [
    {"name": "tests_only", "tools": ["pytest"]},
    {"name": "static_only", "tools": ["ruff", "mypy"]},
    {"name": "security_only", "tools": ["bandit"]},
    {"name": "full_feedback", "tools": ["pytest", "ruff", "mypy", "bandit"]},
]


def main():
    config_path = Path("config/settings.yaml")
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
        logger.error("No tasks loaded.")
        sys.exit(1)

    gen_cfg = config.get("generation", {})
    eiecc_cfg = config.get("feedback", {})
    output_dir = Path(config.get("output", {}).get("data_dir", "data"))

    logger.info("Generating baseline (shared across ablations)...")
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    baseline: list[Generation] = []
    for task in tasks:
        prompt = build_generation_prompt(task, "direct")
        for sample_id in range(gen_cfg.get("samples_per_task", 10)):
            try:
                result = client.generate(prompt=prompt)
                code = result["code"]
                gen = Generation(
                    task_id=task.task_id, benchmark=task.benchmark,
                    model=llm_cfg.get("model", "deepseek-chat"),
                    prompt_variant="direct", sample_id=sample_id,
                    generated_code=code,
                    generated_loc=len(code.splitlines()),
                    generated_tokens=result["completion_tokens"],
                    accepted_loc=len(code.splitlines()),
                    generation_time_s=result["wall_time_s"],
                    condition=Condition.BASELINE,
                )
                baseline.append(gen)
                save_generated_code(code, task.task_id, sample_id, generated_dir)
            except Exception as e:
                logger.error("Baseline gen failed %s/%d: %s", task.task_id, sample_id, e)

    ablation_rows = []
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for ab in ABLATION_CONFIGS:
        logger.info("Running ablation: %s (tools=%s)", ab["name"], ab["tools"])
        try:
            iterations = run_feedback_loop(
                baseline_generations=baseline,
                tasks=tasks,
                llm_client=client,
                max_repair_iterations=eiecc_cfg.get("max_repair_iterations", 3),
                tools=ab["tools"],
                output_dir=output_dir,
            )
            comparison = compare_baseline_vs_treatment(
                baseline_generations=baseline,
                repair_iterations=iterations,
                output_dir=results_dir,
            )
            ablation_rows.append({
                "Feedback Source": ab["name"],
                "Pass Rate (baseline)": comparison.get("baseline_pass_rate", 0),
                "Pass Rate (treatment)": comparison.get("treatment_pass_rate", 0),
                "Pass Rate Improvement": comparison.get("pass_rate_improvement", 0),
                "SW Defect Reduction": comparison.get("sw_defect_reduction", 0),
                "Static Warning Reduction": comparison.get("static_warning_reduction", 0),
                "Token Cost": (
                    comparison.get("total_prompt_tokens", 0)
                    + comparison.get("total_completion_tokens", 0)
                ),
                "Time Cost (s)": comparison.get("total_wall_time_s", 0),
            })
        except Exception as e:
            logger.error("Ablation %s failed: %s", ab["name"], e)
            ablation_rows.append({
                "Feedback Source": ab["name"],
                "Pass Rate (baseline)": "ERR",
                "Pass Rate (treatment)": str(e),
            })

    ablation_csv = results_dir / "ablation_results.csv"
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        if ablation_rows:
            w = csv.DictWriter(f, fieldnames=ablation_rows[0].keys())
            w.writeheader()
            w.writerows(ablation_rows)
    logger.info("Ablation results saved to %s", ablation_csv)


if __name__ == "__main__":
    main()
