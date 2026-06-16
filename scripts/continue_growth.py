#!/usr/bin/env python
"""Continue growth pipeline from existing generated .py files, skipping LLM re-generation."""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgebench.common.schemas import Generation, Condition
from bgebench.common.task_loader import load_tasks, filter_tasks
from bgebench.common.storage import save_generations_csv
from bgebench.growth.run_verification import verify_generations
from bgebench.growth.analyze_growth import analyze_growth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("continue_growth")


def _parse_filename(filename: str) -> tuple[str, int]:
    stem = filename.replace(".py", "")
    if stem.count("_") >= 2:
        parts = stem.rsplit("_", 1)
        task_id = parts[0].replace("_", "/", 1)
        sample_id = int(parts[1])
        return task_id, sample_id
    raise ValueError(f"Cannot parse filename: {filename}")


def reconstruct_generations(generated_dir: Path, tasks: list) -> list[Generation]:
    task_map = {t.task_id: t for t in tasks}
    generations: list[Generation] = []
    py_files = sorted(generated_dir.glob("*.py"))

    for fp in py_files:
        try:
            task_id, sample_id = _parse_filename(fp.name)
        except ValueError:
            logger.warning("Skipping unparseable file: %s", fp.name)
            continue

        if task_id not in task_map:
            logger.warning("Task %s not in loaded tasks, skipping", task_id)
            continue

        code = fp.read_text(encoding="utf-8")
        loc = len(code.splitlines())

        gen = Generation(
            task_id=task_id,
            benchmark=task_map[task_id].benchmark,
            model="deepseek-chat",
            prompt_variant="direct",
            sample_id=sample_id,
            generated_code=code,
            generated_loc=loc,
            generated_tokens=0,
            accepted_loc=loc,
            generation_time_s=0.0,
            condition=Condition.BASELINE,
        )
        generations.append(gen)

    logger.info("Reconstructed %d generations from %d files", len(generations), len(py_files))
    return generations


def main():
    config_path = Path("config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bm = config.get("benchmarks", {})
    tasks = load_tasks(
        humaneval_path=bm.get("humaneval_path"),
        mbpp_path=None,
    )
    tasks = filter_tasks(tasks, max_tasks=config.get("generation", {}).get("max_tasks"))

    output_dir = Path(config.get("output", {}).get("data_dir", "data"))
    generated_dir = output_dir / "generated"

    logger.info("Reconstructing generations from existing files...")
    generations = reconstruct_generations(generated_dir, tasks)
    if not generations:
        logger.error("No generations reconstructed. Aborting.")
        sys.exit(1)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_generations_csv(generations, results_dir / "generations.csv")

    logger.info("Step 2/3: Running verification tools...")
    ver_cfg = config.get("verification", {})
    all_tool_results, all_defects, defects_per_sample = verify_generations(
        generations=generations,
        tasks=tasks,
        tools=ver_cfg.get("tools"),
        output_dir=output_dir,
    )

    logger.info("Step 3/3: Analyzing BGE growth models...")
    generations_df = pd.read_csv(output_dir / "results" / "generations_merged.csv")
    defects_df = pd.read_csv(output_dir / "results" / "defects.csv")

    results = analyze_growth(
        generations_df=generations_df,
        defects_df=defects_df,
        output_dir=output_dir / "results",
    )

    best_bge = results.get("power_law", {}).get("bge", "N/A")
    logger.info("Growth analysis pipeline complete. BGE = %s", best_bge)


if __name__ == "__main__":
    main()
