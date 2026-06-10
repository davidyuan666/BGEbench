#!/usr/bin/env python
"""Reliability Assessment pipeline."""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bgebench.reliability.run_assessment import assess_reliability
from bgebench.reliability.sensitivity import run_sensitivity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reliability")


def main():
    config_path = Path("config/settings.yaml")
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config.get("output", {}).get("data_dir", "data"))
    results_dir = output_dir / "results"

    generations_path = results_dir / "generations.csv"
    defects_path = results_dir / "defects.csv"

    if not generations_path.exists():
        logger.error(
            "Growth measurement output not found at %s. Run 'python scripts/run_growth.py' first.",
            generations_path,
        )
        sys.exit(1)

    generations_df = pd.read_csv(generations_path)
    defects_df = pd.read_csv(defects_path) if defects_path.exists() else pd.DataFrame(
        columns=["task_id", "sample_id", "category"]
    )

    reliability_cfg = config.get("reliability", {})
    lambda1 = reliability_cfg.get("lambda1", 1.0)
    lambda2 = reliability_cfg.get("lambda2", 0.2)
    lambda3 = reliability_cfg.get("lambda3", 1.0)

    logger.info("Step 1/2: Computing RRS for all artifacts...")
    results = assess_reliability(
        generations_df=generations_df,
        defects_df=defects_df,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        output_dir=results_dir,
    )

    logger.info("Step 2/2: Running sensitivity analysis...")
    sensitivity_rows = run_sensitivity(
        generations_df=generations_df,
        defects_df=defects_df,
        output_dir=results_dir,
    )

    counts = {"accept": 0, "review": 0, "reject": 0}
    for r in results:
        counts[r.decision.value] += 1
    logger.info(
        "Reliability assessment pipeline complete. accept=%d, review=%d, reject=%d",
        counts["accept"], counts["review"], counts["reject"],
    )


if __name__ == "__main__":
    main()
