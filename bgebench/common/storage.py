import csv
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from bgebench.common.schemas import (
    Defect,
    Generation,
    ToolResult,
    RepairIteration,
    ReliabilityResult,
    FeedbackItem,
)

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_generations_csv(generations: list[Generation], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task_id", "benchmark", "model", "prompt_variant", "sample_id",
        "generated_loc", "generated_tokens", "accepted_loc",
        "generation_time_s", "condition",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in generations:
            writer.writerow({
                "task_id": g.task_id,
                "benchmark": g.benchmark,
                "model": g.model,
                "prompt_variant": g.prompt_variant,
                "sample_id": g.sample_id,
                "generated_loc": g.generated_loc,
                "generated_tokens": g.generated_tokens,
                "accepted_loc": g.accepted_loc,
                "generation_time_s": g.generation_time_s,
                "condition": g.condition.value,
            })
    logger.info("Saved %d generations to %s", len(generations), path)


def load_generations_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_tool_results_jsonl(results: list[ToolResult], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "task_id": r.task_id,
                "sample_id": r.sample_id,
                "tool": r.tool,
                "exit_code": r.exit_code,
                "summary": r.summary,
                "raw_output_path": r.raw_output_path,
                "passed": r.passed,
                "failed": r.failed,
                "warnings": r.warnings,
                "errors": r.errors,
            }) + "\n")
    logger.info("Saved %d tool results to %s", len(results), path)


def load_tool_results_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def save_defects_csv(defects: list[Defect], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task_id", "sample_id", "category", "description",
        "location", "severity", "tool_source",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in defects:
            writer.writerow({
                "task_id": d.task_id,
                "sample_id": d.sample_id,
                "category": d.category.value,
                "description": d.description,
                "location": d.location,
                "severity": d.severity.value,
                "tool_source": d.tool_source,
            })
    logger.info("Saved %d defects to %s", len(defects), path)


def save_feedback_jsonl(items: list[FeedbackItem], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps({
                "defect_type": item.defect_type,
                "location": item.location,
                "evidence": item.evidence,
                "severity": item.severity,
                "repair_instruction": item.repair_instruction,
            }) + "\n")
    logger.info("Saved %d feedback items to %s", len(items), path)


def save_repair_iterations_csv(iterations: list[RepairIteration], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task_id", "benchmark", "model", "prompt_variant", "sample_id",
        "condition", "iteration", "generated_tokens",
        "loc", "pytest_failed", "ruff_warnings", "mypy_warnings",
        "bandit_warnings", "severity_weighted_defects", "hard_pass",
        "wall_time_s", "prompt_tokens", "completion_tokens",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in iterations:
            writer.writerow({
                "task_id": it.task_id,
                "benchmark": it.benchmark,
                "model": it.model,
                "prompt_variant": it.prompt_variant,
                "sample_id": it.sample_id,
                "condition": it.condition.value,
                "iteration": it.iteration,
                "generated_tokens": it.generated_tokens,
                "loc": it.loc,
                "pytest_failed": it.pytest_failed,
                "ruff_warnings": it.ruff_warnings,
                "mypy_warnings": it.mypy_warnings,
                "bandit_warnings": it.bandit_warnings,
                "severity_weighted_defects": it.severity_weighted_defects,
                "hard_pass": it.hard_pass,
                "wall_time_s": it.wall_time_s,
                "prompt_tokens": it.prompt_tokens,
                "completion_tokens": it.completion_tokens,
            })
    logger.info("Saved %d repair iterations to %s", len(iterations), path)


def save_reliability_csv(results: list[ReliabilityResult], path: Path) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task_id", "benchmark", "model", "prompt_variant", "sample_id",
        "generated_loc", "generated_tokens",
        "defects_raw", "defects_weighted", "repair_iterations",
        "final_failure", "pass_rate_public", "pass_rate_hidden",
        "rrs", "decision", "decision_threshold_version",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "task_id": r.task_id,
                "benchmark": r.benchmark,
                "model": r.model,
                "prompt_variant": r.prompt_variant,
                "sample_id": r.sample_id,
                "generated_loc": r.generated_loc,
                "generated_tokens": r.generated_tokens,
                "defects_raw": r.defects_raw,
                "defects_weighted": r.defects_weighted,
                "repair_iterations": r.repair_iterations,
                "final_failure": r.final_failure,
                "pass_rate_public": r.pass_rate_public,
                "pass_rate_hidden": r.pass_rate_hidden,
                "rrs": r.rrs,
                "decision": r.decision.value,
                "decision_threshold_version": r.decision_threshold_version,
            })
    logger.info("Saved %d reliability results to %s", len(results), path)


def save_generated_code(code: str, task_id: str, sample_id: int, base_dir: Path) -> Path:
    ensure_dir(base_dir)
    safe_id = task_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
    filepath = base_dir / f"{safe_id}_{sample_id}.py"
    filepath.write_text(code, encoding="utf-8")
    return filepath


def load_generated_code(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_merged_generations_csv(
    generations: list[Generation],
    defects_per_sample: dict[tuple[str, int], dict[str, int]],
    path: Path,
) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "task_id", "benchmark", "model", "prompt_variant", "sample_id",
        "generated_loc", "generated_tokens", "accepted_loc",
        "tests_total", "tests_passed", "tests_failed",
        "syntax_errors", "runtime_errors", "semantic_defects", "boundary_defects",
        "security_warnings", "api_errors", "maintainability_warnings",
        "confirmed_defects", "defect_density_kloc", "generation_time_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for g in generations:
            key = (g.task_id, g.sample_id)
            defects = defects_per_sample.get(key, {})
            total_defects = sum(defects.values())
            row = {
                "task_id": g.task_id,
                "benchmark": g.benchmark,
                "model": g.model,
                "prompt_variant": g.prompt_variant,
                "sample_id": g.sample_id,
                "generated_loc": g.generated_loc,
                "generated_tokens": g.generated_tokens,
                "accepted_loc": g.accepted_loc,
                "tests_total": defects.get("tests_total", 0),
                "tests_passed": defects.get("tests_passed", 0),
                "tests_failed": defects.get("tests_failed", 0),
                "syntax_errors": defects.get("syntax", 0),
                "runtime_errors": defects.get("runtime", 0),
                "semantic_defects": defects.get("semantic", 0),
                "boundary_defects": defects.get("boundary", 0),
                "security_warnings": defects.get("security", 0),
                "api_errors": defects.get("api", 0),
                "maintainability_warnings": defects.get("maintainability", 0),
                "confirmed_defects": total_defects,
                "defect_density_kloc": round(
                    (total_defects / (g.generated_loc / 1000.0)) if g.generated_loc > 0 else 0.0, 2
                ),
                "generation_time_s": g.generation_time_s,
            }
            writer.writerow(row)
    logger.info("Saved merged generations + defects to %s", path)
