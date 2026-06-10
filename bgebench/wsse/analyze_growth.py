import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


def analyze_growth(
    generations_df: pd.DataFrame,
    defects_df: pd.DataFrame,
    output_dir: Path = Path("data/results"),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    defect_counts = (
        defects_df.groupby(["task_id", "sample_id"])
        .size()
        .reset_index(name="defect_count")
    )
    merged = generations_df.merge(defect_counts, on=["task_id", "sample_id"], how="left")
    merged["defect_count"] = merged["defect_count"].fillna(0).astype(int)
    merged["log_V"] = np.log(merged["generated_loc"].values + 1)
    merged["log_B"] = np.log(merged["defect_count"].values + 1)

    V = merged["generated_loc"].values
    B = merged["defect_count"].values
    log_V = merged["log_V"].values
    log_B = merged["log_B"].values

    results = {}

    results["descriptive"] = {
        "n_samples": len(merged),
        "n_tasks": merged["task_id"].nunique(),
        "mean_loc": float(np.mean(V)),
        "std_loc": float(np.std(V)),
        "mean_defects": float(np.mean(B)),
        "std_defects": float(np.std(B)),
    }

    linear_slope, linear_intercept, r_value, p_value, std_err = stats.linregress(V, B)
    linear_aic, linear_bic = _aic_bic_linear(B, linear_intercept + linear_slope * V)
    results["linear"] = {
        "model": "B = a + b*V",
        "intercept": float(linear_intercept),
        "slope": float(linear_slope),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "aic": float(linear_aic),
        "bic": float(linear_bic),
        "bge_estimate": "slope-based (not log-log)",
        "interpretation": _interpret_slope(linear_slope),
    }

    log_slope, log_intercept, log_r, log_p, log_se = stats.linregress(np.log(V + 1), B)
    log_pred = log_intercept + log_slope * np.log(V + 1)
    log_aic, log_bic = _aic_bic_linear(B, log_pred)
    results["logarithmic"] = {
        "model": "B = a + b*log(V+1)",
        "intercept": float(log_intercept),
        "log_coef": float(log_slope),
        "r_squared": float(log_r ** 2),
        "p_value": float(log_p),
        "aic": float(log_aic),
        "bic": float(log_bic),
        "bge_estimate": f"local elasticity ≈ {log_slope:.4f}",
        "interpretation": _interpret_elasticity(log_slope),
    }

    power_slope, power_intercept, power_r, power_p, power_se = stats.linregress(log_V, log_B)
    power_pred = np.exp(power_intercept) * (V ** power_slope)
    power_aic, power_bic = _aic_bic_nonlinear(B, power_pred)
    results["power_law"] = {
        "model": "log(B+1) = alpha + beta*log(V+1)",
        "alpha": float(power_intercept),
        "beta": float(power_slope),
        "r_squared": float(power_r ** 2),
        "p_value": float(power_p),
        "aic": float(power_aic),
        "bic": float(power_bic),
        "bge": float(power_slope),
        "bge_ci": _confidence_interval(power_slope, power_se, len(V)),
        "interpretation": _interpret_elasticity(power_slope),
    }

    try:
        nb_fit = _fit_negative_binomial(B, V)
        results["negative_binomial"] = nb_fit
    except Exception as e:
        logger.warning("Negative binomial regression failed: %s", e)
        results["negative_binomial"] = {"error": str(e)}

    results["model_ranking"] = _rank_models(results)

    results["defect_taxonomy"] = _taxonomy_summary(defects_df)

    results["prompt_sensitivity"] = _prompt_sensitivity(generations_df, defects_df)

    _save_results_table(results, output_dir)

    logger.info("BGE analysis complete. Best model beta=%.4f", power_slope)
    return results


def _aic_bic_linear(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    n = len(y)
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    k = 2
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic


def _aic_bic_nonlinear(y: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    return _aic_bic_linear(y, y_pred)


def _confidence_interval(beta: float, se: float, n: int) -> str:
    if n <= 2:
        return "N/A"
    t_crit = stats.t.ppf(0.975, n - 2)
    lo = beta - t_crit * se
    hi = beta + t_crit * se
    return f"[{lo:.4f}, {hi:.4f}]"


def _interpret_elasticity(beta: float) -> str:
    if beta <= 0:
        return "No defect growth with volume"
    elif beta < 0.5:
        return "Strongly sublinear: defect risk grows slowly"
    elif beta < 0.8:
        return "Sublinear: moderate defect growth"
    elif beta <= 1.2:
        return "Approximately linear: proportional defect growth"
    else:
        return "Superlinear: accelerating defect risk—caution advised"


def _interpret_slope(slope: float) -> str:
    if slope < 0:
        return "Negative slope: more LOC fewer defects (unusual)"
    elif slope < 0.01:
        return "Near-zero: negligible defect increase"
    elif slope < 0.1:
        return f"Moderate: ~{slope:.3f} defects per LOC"
    else:
        return f"High: ~{slope:.3f} defects per LOC"


def _fit_negative_binomial(B: np.ndarray, V: np.ndarray) -> dict:
    import statsmodels.api as sm

    X = sm.add_constant(V)
    model = sm.GLM(B, X, family=sm.families.NegativeBinomial())
    result = model.fit()
    return {
        "model": "Negative Binomial (B ~ V)",
        "intercept": float(result.params[0]),
        "coef_V": float(result.params[1]),
        "pseudo_r2": 1 - result.deviance / result.null_deviance,
        "aic": float(result.aic),
        "bic": float(result.bic_llf if hasattr(result, "bic_llf") else result.aic),
        "interpretation": _interpret_elasticity(result.params[1]),
    }


def _rank_models(results: dict) -> list[dict]:
    models = []
    for key in ["linear", "logarithmic", "power_law", "negative_binomial"]:
        entry = results.get(key, {})
        if "error" in entry:
            continue
        aic = entry.get("aic", float("inf"))
        bic = entry.get("bic", float("inf"))
        r2 = entry.get("r_squared", entry.get("pseudo_r2", 0))
        models.append({"model": entry.get("model", key), "aic": aic, "bic": bic, "r_squared": r2})
    models.sort(key=lambda m: m["aic"])
    return models


def _taxonomy_summary(defects_df: pd.DataFrame) -> dict[str, Any]:
    if defects_df.empty:
        return {"total_defects": 0, "categories": {}}
    counts = defects_df["category"].value_counts().to_dict()
    total = int(defects_df["category"].count())
    return {
        "total_defects": total,
        "categories": counts,
    }


def _prompt_sensitivity(
    gen_df: pd.DataFrame, defect_df: pd.DataFrame
) -> list[dict]:
    dc = (
        defect_df.groupby(["task_id", "sample_id"])
        .size()
        .reset_index(name="defect_count")
    )
    merged = gen_df.merge(dc, on=["task_id", "sample_id"], how="left")
    merged["defect_count"] = merged["defect_count"].fillna(0)
    rows = []
    for pv in merged["prompt_variant"].unique():
        subset = merged[merged["prompt_variant"] == pv]
        if len(subset) == 0:
            continue
        logV = np.log(subset["generated_loc"].values + 1)
        logB = np.log(subset["defect_count"].values + 1)
        if np.std(logV) < 1e-9:
            continue
        slope, _, _, _, _ = stats.linregress(logV, logB)
        rows.append({
            "prompt_variant": pv,
            "mean_loc": float(subset["generated_loc"].mean()),
            "mean_defects": float(subset["defect_count"].mean()),
            "bge": float(slope),
        })
    return rows


def _save_results_table(results: dict, output_dir: Path) -> None:
    main_table_path = output_dir / "bge_fit_results.csv"
    main_rows = []
    for key in ["linear", "logarithmic", "power_law", "negative_binomial"]:
        entry = results.get(key, {})
        if not entry or "error" in entry:
            continue
        main_rows.append({
            "Model": entry.get("model", key),
            "BGE": entry.get("bge", entry.get("bge_estimate", "")),
            "R²": entry.get("r_squared", entry.get("pseudo_r2", "")),
            "AIC": entry.get("aic", ""),
            "BIC": entry.get("bic", ""),
            "Interpretation": entry.get("interpretation", ""),
        })
    with open(main_table_path, "w", newline="", encoding="utf-8") as f:
        if main_rows:
            w = csv.DictWriter(f, fieldnames=main_rows[0].keys())
            w.writeheader()
            w.writerows(main_rows)
    logger.info("BGE fit results saved to %s", main_table_path)
