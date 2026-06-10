import time
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from bgebench.common.llm_client import LLMClient
from bgebench.common.schemas import Generation, RepairIteration, Task
from bgebench.common.verification import run_all_tools
from bgebench.common.defect_classifier import classify_defects, compute_severity_weighted
from bgebench.common.storage import save_repair_iterations_csv, save_generated_code
from bgebench.eiecc.normalize_feedback import normalize_feedback, build_repair_prompt

logger = logging.getLogger(__name__)


def run_feedback_loop(
    baseline_generations: list[Generation],
    tasks: list[Task],
    llm_client: LLMClient,
    max_repair_iterations: int = 3,
    tools: Optional[list[str]] = None,
    output_dir: Path = Path("data"),
) -> list[RepairIteration]:
    task_map = {t.task_id: t for t in tasks}
    repaired_dir = output_dir / "repaired"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_iterations: list[RepairIteration] = []

    with tqdm(total=len(baseline_generations), desc="Repair loop") as pbar:
        for gen in baseline_generations:
            task = task_map.get(gen.task_id)
            if task is None:
                logger.warning("Task %s not found, skipping repair", gen.task_id)
                pbar.update(1)
                continue

            current_code = gen.generated_code
            for iteration in range(max_repair_iterations + 1):
                start_time = time.time()

                tool_results = run_all_tools(
                    code=current_code,
                    task=task,
                    sample_id=gen.sample_id,
                    tools=tools,
                    tmp_dir=log_dir,
                )
                defects = classify_defects(tool_results)
                sw_defects = compute_severity_weighted(defects)

                pytest_failed = sum(tr.failed for tr in tool_results if tr.tool == "pytest")
                ruff_warnings = sum(tr.warnings for tr in tool_results if tr.tool == "ruff")
                mypy_warnings = sum(tr.warnings for tr in tool_results if tr.tool == "mypy")
                bandit_warnings = sum(tr.warnings for tr in tool_results if tr.tool == "bandit")

                hard_pass = sw_defects == 0 and pytest_failed == 0

                loc = len(current_code.splitlines())

                iter_record = RepairIteration(
                    task_id=gen.task_id,
                    sample_id=gen.sample_id,
                    iteration=iteration,
                    code=current_code,
                    loc=loc,
                    pytest_failed=pytest_failed,
                    ruff_warnings=ruff_warnings,
                    mypy_warnings=mypy_warnings,
                    bandit_warnings=bandit_warnings,
                    severity_weighted_defects=sw_defects,
                    hard_pass=hard_pass,
                    wall_time_s=time.time() - start_time,
                    prompt_tokens=0,
                    completion_tokens=0,
                )
                all_iterations.append(iter_record)

                save_generated_code(
                    current_code, gen.task_id,
                    gen.sample_id * 100 + iteration, repaired_dir,
                )

                if hard_pass or iteration >= max_repair_iterations:
                    break

                if defects:
                    feedback_items = normalize_feedback(tool_results, defects)
                    repair_prompt = build_repair_prompt(current_code, feedback_items)
                    try:
                        repair_result = llm_client.generate(
                            prompt=repair_prompt,
                            system="You are a code repair assistant. Return only corrected code, no explanation.",
                        )
                        current_code = repair_result["code"]
                        all_iterations[-1].prompt_tokens = repair_result["prompt_tokens"]
                        all_iterations[-1].completion_tokens = repair_result["completion_tokens"]
                    except Exception as e:
                        logger.error(
                            "Repair failed for %s sample %d iter %d: %s",
                            gen.task_id, gen.sample_id, iteration, e,
                        )
                        break

            pbar.set_postfix(
                task=gen.task_id[:20],
                final_pass=all_iterations[-1].hard_pass,
            )
            pbar.update(1)

    csv_path = output_dir / "results" / "feedback_iterations.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    save_repair_iterations_csv(all_iterations, csv_path)

    logger.info("Repair loop complete: %d iteration records", len(all_iterations))
    return all_iterations
