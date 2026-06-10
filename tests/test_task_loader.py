import json
import tempfile
from pathlib import Path

from bgebench.common.task_loader import (
    load_humaneval,
    load_mbpp,
    load_repair_tasks,
    build_generation_prompt,
    filter_tasks,
)
from bgebench.common.schemas import Task


def test_load_humaneval():
    data = json.dumps({
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b",
        "test": "def test():\n    assert add(1,2)==3",
        "entry_point": "add",
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(data + "\n")
        f.flush()
        tasks = load_humaneval(f.name)
    Path(f.name).unlink(missing_ok=True)

    assert len(tasks) == 1
    assert tasks[0].task_id == "HumanEval/0"
    assert tasks[0].benchmark == "HumanEval"
    assert tasks[0].entry_point == "add"


def test_load_mbpp():
    data = json.dumps({
        "task_id": 1,
        "text": "Write a function to add two numbers.",
        "code": "def add(a,b): return a+b",
        "test_list": ["add(1,2)==3"],
        "test_setup_code": "",
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(data + "\n")
        f.flush()
        tasks = load_mbpp(f.name)
    Path(f.name).unlink(missing_ok=True)

    assert len(tasks) == 1
    assert tasks[0].benchmark == "MBPP"
    assert "assert add(1,2)==3" in tasks[0].test_code


def test_load_repair_tasks():
    data = json.dumps([{
        "task_id": "Repair/1",
        "benchmark": "repair",
        "prompt": "Fix the bug",
        "canonical_solution": "pass",
        "test_code": "assert True",
    }])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(data)
        f.flush()
        tasks = load_repair_tasks(f.name)
    Path(f.name).unlink(missing_ok=True)

    assert len(tasks) == 1
    assert tasks[0].task_id == "Repair/1"


def test_build_generation_prompt_direct():
    task = Task(task_id="test", benchmark="test", prompt="Add two numbers")
    prompt = build_generation_prompt(task, "direct")
    assert "Add two numbers" in prompt
    assert "Return only the Python code" in prompt


def test_build_generation_prompt_concise():
    task = Task(task_id="test", benchmark="test", prompt="Add two numbers")
    prompt = build_generation_prompt(task, "concise")
    assert "Add two numbers" in prompt
    assert "Output only code" in prompt


def test_build_generation_prompt_test_aware():
    task = Task(
        task_id="test", benchmark="test", prompt="Add two numbers",
        test_code="assert True",
    )
    prompt = build_generation_prompt(task, "test_aware")
    assert "Add two numbers" in prompt
    assert "input-output contract" in prompt


def test_filter_tasks_by_max():
    tasks = [Task(task_id=str(i), benchmark="t", prompt="p") for i in range(10)]
    filtered = filter_tasks(tasks, max_tasks=5)
    assert len(filtered) == 5


def test_filter_tasks_by_ids():
    tasks = [Task(task_id=str(i), benchmark="t", prompt="p") for i in range(5)]
    filtered = filter_tasks(tasks, task_ids=["1", "3"])
    assert len(filtered) == 2
