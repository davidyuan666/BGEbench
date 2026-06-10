import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from bgebench.common.schemas import (
    Decision,
    DefectCategory,
    ReliabilityResult,
)
from bgebench.common.defect_classifier import DEFECT_WEIGHTS
from bgebench.common.storage import save_reliability_csv

logger = logging.getLogger(__name__)


def assess_reliability(
    generations_df: pd.DataFrame,
    defects_df: pd.DataFrame,
    lambda1: float = 1.0,
    lambda2: float = 0.2,
    lambda3: float = 1.0,
    rrs_accept_threshold: float = 2.0,
    rrs_reject_threshold: float = 5.0,
    output_dir: Path = Path("data/results"),
    config_version: str = "default",
) -> list[ReliabilityResult]:
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_summary = (
        defects_df.groupby(["task_id", "sample_id"])
        .agg(
            defects_raw=("category", "count"),
            defects_weighted=("category", lambda cats: sum(DEFECT_WEIGHTS.get(DefectCategory(c), 1) for c in cats)),
        )
        .reset_index()
    )

    merged = generations_df.merge(defect_summary, on=["task_id", "sample_id"], how="left")
    merged["defects_raw"] = merged["defects_raw"].fillna(0).astype(int)
    merged["defects_weighted"] = merged["defects_weighted"].fillna(0.0)

    results: list[ReliabilityResult] = []
    for _, row in merged.iterrows():
        V = max(row.get("generated_loc", 0), 0)
        B_w = float(row["defects_weighted"])
        R_val = 0
        F_val = 1.0 if B_w > 0 else 0.0

        pass_rate = 0.0
        if V > 0:
            pass_rate = max(0.0, 1.0 - (B_w / max(V, 1)))

        rrs = lambda1 * (B_w / (V + 1)) + lambda2 * R_val + lambda3 * F_val

        if rrs <= rrs_accept_threshold and F_val == 0:
            decision = Decision.ACCEPT
        elif rrs >= rrs_reject_threshold or F_val == 1:
            decision = Decision.REJECT
        else:
            decision = Decision.REVIEW

        results.append(
            ReliabilityResult(
                task_id=str(row["task_id"]),
                benchmark=str(row.get("benchmark", "")),
                model=str(row.get("model", "")),
                prompt_variant=str(row.get("prompt_variant", "")),
                sample_id=int(row["sample_id"]),
                generated_loc=int(V),
                generated_tokens=int(row.get("generated_tokens", 0)),
                defects_raw=int(row["defects_raw"]),
                defects_weighted=round(B_w, 4),
                repair_iterations=int(R_val),
                final_failure=bool(F_val),
                pass_rate_public=round(pass_rate, 4),
                pass_rate_hidden=-1.0,
                rrs=round(rrs, 4),
                decision=decision,
                decision_threshold_version=config_version,
            )
        )

    save_reliability_csv(results, output_dir / "reliability_assessment.csv")

    rrs_config = {
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "rrs_accept_threshold": rrs_accept_threshold,
        "rrs_reject_threshold": rrs_reject_threshold,
        "rrs_formula": "RRS = lambda1 * B_w/(V+1) + lambda2 * R + lambda3 * F",
        "defect_weights": {k.value: v for k, v in DEFECT_WEIGHTS.items()},
    }
    with open(output_dir / "rrs_config.json", "w", encoding="utf-8") as f:
        json.dump(rrs_config, f, indent=2)

    counts = {"accept": 0, "review": 0, "reject": 0}
    for r in results:
        counts[r.decision.value] += 1
    logger.info(
        "ICRE assessment complete: %d results (accept=%d, review=%d, reject=%d)",
        len(results), counts["accept"], counts["review"], counts["reject"],
    )
    return results
