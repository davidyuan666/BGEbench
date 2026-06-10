import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from bgebench.common.schemas import Defect, Generation, Task, ToolResult
from bgebench.common.verification import run_all_tools
from bgebench.common.defect_classifier import classify_defects, count_by_category, compute_severity_weighted
from bgebench.common.storage import save_tool_results_jsonl, save_defects_csv

logger = logging.getLogger(__name__)


def verify_generations(
    generations: list[Generation],
    tasks: list[Task],
    tools: Optional[list[str]] = None,
    output_dir: Path = Path("data"),
) -> tuple[list[ToolResult], list[Defect]]:
    task_map = {t.task_id: t for t in tasks}
    all_tool_results: list[ToolResult] = []
    all_defects: list[Defect] = []

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=len(generations), desc="Verifying") as pbar:
        for gen in generations:
            task = task_map.get(gen.task_id)
            if task is None:
                logger.warning("Task %s not found, skipping verification", gen.task_id)
                pbar.update(1)
                continue

            try:
                tool_results = run_all_tools(
                    code=gen.generated_code,
                    task=task,
                    sample_id=gen.sample_id,
                    tools=tools,
                    tmp_dir=log_dir,
                )
                all_tool_results.extend(tool_results)

                defects = classify_defects(tool_results)
                all_defects.extend(defects)

                cat_counts = count_by_category(defects)
                pbar.set_postfix(
                    task=gen.task_id[:20],
                    defects=len(defects),
                    cats="/".join(f"{k}:{v}" for k, v in cat_counts.items()),
                )
            except Exception as e:
                logger.error(
                    "Verification failed for %s sample %d: %s",
                    gen.task_id, gen.sample_id, e,
                )

            pbar.update(1)

    tool_path = output_dir / "results" / "tool_outputs.jsonl"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    save_tool_results_jsonl(all_tool_results, tool_path)

    defect_path = output_dir / "results" / "defects.csv"
    save_defects_csv(all_defects, defect_path)

    logger.info(
        "WSSE verification complete: %d tool results, %d defects",
        len(all_tool_results), len(all_defects),
    )
    return all_tool_results, all_defects
