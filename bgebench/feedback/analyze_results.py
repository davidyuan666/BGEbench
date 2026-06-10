import csv
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from bgebench.common.schemas import Generation, RepairIteration

logger = logging.getLogger(__name__)


def compare_baseline_vs_treatment(
    baseline_generations: list[Generation],
    repair_iterations: list[RepairIteration],
    output_dir: Path = Path("data/results"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    iter_df = pd.DataFrame([{
        "task_id": r.task_id,
        "sample_id": r.sample_id,
        "iteration": r.iteration,
        "loc": r.loc,
        "pytest_failed": r.pytest_failed,
        "ruff_warnings": r.ruff_warnings,
        "mypy_warnings": r.mypy_warnings,
        "bandit_warnings": r.bandit_warnings,
        "severity_weighted_defects": r.severity_weighted_defects,
        "hard_pass": r.hard_pass,
        "wall_time_s": r.wall_time_s,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
    } for r in repair_iterations])

    if iter_df.empty:
        logger.warning("No repair iteration data available")
        return {}

    final_iter = iter_df.loc[iter_df.groupby(["task_id", "sample_id"])["iteration"].idxmax()].copy()

    baseline_pass_count = final_iter[final_iter["iteration"] == 0]["hard_pass"].sum()
    total_samples = len(baseline_generations)
    baseline_pass_rate = baseline_pass_count / total_samples if total_samples else 0

    treatment_pass_count = final_iter["hard_pass"].sum()
    treatment_pass_rate = treatment_pass_count / total_samples if total_samples else 0

    baseline_sw_defects = final_iter[final_iter["iteration"] == 0]["severity_weighted_defects"].mean()
    treatment_sw_defects = final_iter["severity_weighted_defects"].mean()

    baseline_static = (
        final_iter[final_iter["iteration"] == 0][
            ["ruff_warnings", "mypy_warnings", "bandit_warnings"]
        ].sum(axis=1).mean()
    )
    treatment_static = final_iter[["ruff_warnings", "mypy_warnings", "bandit_warnings"]].sum(axis=1).mean()

    total_prompt_tokens = iter_df["prompt_tokens"].sum()
    total_completion_tokens = iter_df["completion_tokens"].sum()
    total_time = iter_df["wall_time_s"].sum()

    comparison_rows = []
    for _, row in final_iter.iterrows():
        baseline_row = final_iter[
            (final_iter["task_id"] == row["task_id"])
            & (final_iter["sample_id"] == row["sample_id"])
            & (final_iter["iteration"] == 0)
        ]
        b_pass = baseline_row["hard_pass"].values[0] if len(baseline_row) > 0 else False
        b_defects = baseline_row["severity_weighted_defects"].values[0] if len(baseline_row) > 0 else 0
        b_static = 0
        if len(baseline_row) > 0:
            b_static = baseline_row[["ruff_warnings", "mypy_warnings", "bandit_warnings"]].sum(axis=1).values[0]

        t_static = (
            row["ruff_warnings"] + row["mypy_warnings"] + row["bandit_warnings"]
        )

        comparison_rows.append({
            "task_id": row["task_id"],
            "sample_id": row["sample_id"],
            "baseline_pass": b_pass,
            "treatment_pass": row["hard_pass"],
            "baseline_weighted_defects": b_defects,
            "treatment_weighted_defects": row["severity_weighted_defects"],
            "baseline_static_warnings": int(b_static),
            "treatment_static_warnings": int(t_static),
            "repair_iterations": int(row["iteration"]),
            "token_cost_total": int(row["prompt_tokens"] + row["completion_tokens"]),
            "time_cost_total_s": round(row["wall_time_s"], 2),
        })

    csv_path = output_dir / "final_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if comparison_rows:
            w = csv.DictWriter(f, fieldnames=comparison_rows[0].keys())
            w.writeheader()
            w.writerows(comparison_rows)

    results = {
        "baseline_pass_rate": round(baseline_pass_rate, 4),
        "treatment_pass_rate": round(treatment_pass_rate, 4),
        "pass_rate_improvement": round(treatment_pass_rate - baseline_pass_rate, 4),
        "baseline_sw_defects": round(baseline_sw_defects, 4),
        "treatment_sw_defects": round(treatment_sw_defects, 4),
        "sw_defect_reduction": round(baseline_sw_defects - treatment_sw_defects, 4),
        "baseline_static_warnings": round(baseline_static, 4),
        "treatment_static_warnings": round(treatment_static, 4),
        "static_warning_reduction": round(baseline_static - treatment_static, 4),
        "total_prompt_tokens": int(total_prompt_tokens),
        "total_completion_tokens": int(total_completion_tokens),
        "total_wall_time_s": round(total_time, 2),
        "samples_total": total_samples,
    }

    feedback_effectiveness_rows = [{
        "condition": "baseline",
        "pass_rate": results["baseline_pass_rate"],
        "weighted_defects": results["baseline_sw_defects"],
        "static_warnings": results["baseline_static_warnings"],
    }, {
        "condition": "treatment",
        "pass_rate": results["treatment_pass_rate"],
        "weighted_defects": results["treatment_sw_defects"],
        "static_warnings": results["treatment_static_warnings"],
    }]
    eff_path = output_dir / "feedback_effectiveness.csv"
    with open(eff_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=feedback_effectiveness_rows[0].keys())
        w.writeheader()
        w.writerows(feedback_effectiveness_rows)

    logger.info(
        "Feedback analysis: pass_rate baseline=%.4f treatment=%.4f improvement=%.4f",
        baseline_pass_rate, treatment_pass_rate, results["pass_rate_improvement"],
    )
    return results
