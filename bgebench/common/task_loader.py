import json
import logging
from pathlib import Path
from typing import Optional

from bgebench.common.schemas import Task

logger = logging.getLogger(__name__)

BENCHMARK_URLS = {
    "humaneval": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
    "mbpp": "https://github.com/google-research/google-research/raw/master/mbpp/mbpp.jsonl",
}


def load_humaneval(path: str) -> list[Task]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            task = Task(
                task_id=data["task_id"],
                benchmark="HumanEval",
                prompt=data["prompt"],
                canonical_solution=data.get("canonical_solution", ""),
                test_code=data.get("test", ""),
                entry_point=data.get("entry_point", ""),
            )
            tasks.append(task)
    logger.info("Loaded %d tasks from HumanEval at %s", len(tasks), path)
    return tasks


def load_mbpp(path: str) -> list[Task]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            task_id = f"Mbpp/{data.get('task_id', len(tasks))}"
            prompt = data.get("text", "")
            test_list = data.get("test_list", [])
            test_setup = data.get("test_setup_code", "")
            test_code = _build_mbpp_test(test_list, test_setup)
            task = Task(
                task_id=task_id,
                benchmark="MBPP",
                prompt=prompt,
                canonical_solution=data.get("code", ""),
                test_code=test_code,
            )
            tasks.append(task)
    logger.info("Loaded %d tasks from MBPP at %s", len(tasks), path)
    return tasks


def load_repair_tasks(path: str) -> list[Task]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        task = Task(
            task_id=item.get("task_id", f"Repair/{len(tasks)}"),
            benchmark=item.get("benchmark", "repair"),
            prompt=item.get("prompt", ""),
            canonical_solution=item.get("canonical_solution", ""),
            test_code=item.get("test_code", ""),
            entry_point=item.get("entry_point", ""),
        )
        tasks.append(task)
    logger.info("Loaded %d repair tasks from %s", len(tasks), path)
    return tasks


def load_tasks(
    humaneval_path: Optional[str] = None,
    mbpp_path: Optional[str] = None,
    repair_path: Optional[str] = None,
) -> list[Task]:
    tasks: list[Task] = []
    if humaneval_path and Path(humaneval_path).exists():
        tasks.extend(load_humaneval(humaneval_path))
    if mbpp_path and Path(mbpp_path).exists():
        tasks.extend(load_mbpp(mbpp_path))
    if repair_path and Path(repair_path).exists():
        tasks.extend(load_repair_tasks(repair_path))
    logger.info("Total tasks loaded: %d", len(tasks))
    return tasks


def filter_tasks(
    tasks: list[Task],
    max_tasks: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
) -> list[Task]:
    if task_ids:
        id_set = set(task_ids)
        tasks = [t for t in tasks if t.task_id in id_set]
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
    return tasks


def build_generation_prompt(
    task: Task, variant: str = "direct"
) -> str:
    if variant == "direct":
        return _direct_prompt(task)
    elif variant == "concise":
        return _concise_prompt(task)
    elif variant == "test_aware":
        return _test_aware_prompt(task)
    else:
        return _direct_prompt(task)


def _direct_prompt(task: Task) -> str:
    return (
        f"Write a Python function that solves the following task.\n\n"
        f"{task.prompt}\n\n"
        f"Return only the Python code, no explanation."
    )


def _concise_prompt(task: Task) -> str:
    return (
        f"Complete the following Python function. Output only code.\n\n"
        f"{task.prompt}"
    )


def _test_aware_prompt(task: Task) -> str:
    base = _direct_prompt(task)
    if task.test_code:
        return (
            f"{base}\n\n"
            f"The code will be tested with assertions. Ensure your function "
            f"handles the declared input-output contract correctly."
        )
    return base


def _build_mbpp_test(test_list: list[str], setup_code: str) -> str:
    lines = []
    if setup_code:
        lines.append(setup_code)
    for test in test_list:
        lines.append(f"assert {test}")
    return "\n".join(lines)
