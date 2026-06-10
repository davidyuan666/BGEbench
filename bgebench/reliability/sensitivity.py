import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bgebench.common.schemas import Decision
from bgebench.reliability.run_assessment import assess_reliability

logger = logging.getLogger(__name__)


def run_sensitivity(
    generations_df: pd.DataFrame,
    defects_df: pd.DataFrame,
    output_dir: Path = Path("data/results"),
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    lambda_configs = [
        {"name": "default", "l1": 1.0, "l2": 0.2, "l3": 1.0},
        {"name": "high_volume_weight", "l1": 2.0, "l2": 0.2, "l3": 1.0},
        {"name": "high_repair_weight", "l1": 1.0, "l2": 0.5, "l3": 1.0},
        {"name": "high_failure_weight", "l1": 1.0, "l2": 0.2, "l3": 2.0},
        {"name": "balanced_low", "l1": 0.5, "l2": 0.1, "l3": 0.5},
    ]

    threshold_configs = [
        {"name": "strict", "accept": 1.0, "reject": 3.0},
        {"name": "default", "accept": 2.0, "reject": 5.0},
        {"name": "lenient", "accept": 4.0, "reject": 8.0},
    ]

    sensitivity_rows: list[dict] = []

    for lc in lambda_configs:
        for tc in threshold_configs:
            config_key = f"{lc['name']}_{tc['name']}"
            try:
                results = assess_reliability(
                    generations_df=generations_df,
                    defects_df=defects_df,
                    lambda1=lc["l1"],
                    lambda2=lc["l2"],
                    lambda3=lc["l3"],
                    rrs_accept_threshold=tc["accept"],
                    rrs_reject_threshold=tc["reject"],
                    output_dir=output_dir,
                    config_version=config_key,
                )
                counts = {"accept": 0, "review": 0, "reject": 0}
                rrs_values = []
                for r in results:
                    counts[r.decision.value] += 1
                    rrs_values.append(r.rrs)

                n = len(results)
                sensitivity_rows.append({
                    "lambda_config": lc["name"],
                    "threshold_config": tc["name"],
                    "lambda1": lc["l1"],
                    "lambda2": lc["l2"],
                    "lambda3": lc["l3"],
                    "accept_threshold": tc["accept"],
                    "reject_threshold": tc["reject"],
                    "n_accept": counts["accept"],
                    "n_review": counts["review"],
                    "n_reject": counts["reject"],
                    "accept_pct": round(counts["accept"] / n * 100, 1) if n else 0,
                    "review_pct": round(counts["review"] / n * 100, 1) if n else 0,
                    "reject_pct": round(counts["reject"] / n * 100, 1) if n else 0,
                    "mean_rrs": round(np.mean(rrs_values), 4) if rrs_values else 0,
                    "median_rrs": round(np.median(rrs_values), 4) if rrs_values else 0,
                })
            except Exception as e:
                logger.error("Sensitivity run %s failed: %s", config_key, e)

    csv_path = output_dir / "sensitivity_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if sensitivity_rows:
            w = csv.DictWriter(f, fieldnames=sensitivity_rows[0].keys())
            w.writeheader()
            w.writerows(sensitivity_rows)
    logger.info("Sensitivity analysis saved to %s (%d configs)", csv_path, len(sensitivity_rows))

    return sensitivity_rows
