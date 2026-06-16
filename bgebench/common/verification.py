import os
import subprocess
import sys
import tempfile
import json
import logging
from pathlib import Path
from typing import Optional

from bgebench.common.schemas import ToolResult, Task

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60

_TOOL_CACHE: dict[str, str] = {}

_SCRIPT_DIRS = [
    os.path.join(os.path.dirname(sys.executable), "Scripts"),
    os.path.join(sys.prefix, "Scripts"),
    os.path.join(sys.base_prefix, "Scripts"),
    os.path.expanduser(r"~\AppData\Roaming\Python\Python314\Scripts"),
]


def _resolve_tool(name: str) -> str:
    if name in _TOOL_CACHE:
        return _TOOL_CACHE[name]
    for d in _SCRIPT_DIRS:
        exe = os.path.join(d, f"{name}.exe")
        if os.path.isfile(exe):
            _TOOL_CACHE[name] = exe
            return exe
    _TOOL_CACHE[name] = name
    return name


def run_pytest(
    code: str,
    test_code: str,
    task_id: str,
    sample_id: int,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
    entry_point: Optional[str] = None,
) -> ToolResult:
    with _temp_dir(tmp_dir) as workdir:
        test_file = workdir / f"test_{_safe_name(task_id)}_{sample_id}.py"
        wrapper = ""
        if entry_point and "def check(" in test_code:
            wrapper = f"\n\ndef test_{entry_point}():\n    check({entry_point})\n"
        test_file.write_text(f"{code}\n\n{test_code}\n{wrapper}", encoding="utf-8")

        try:
            proc = subprocess.run(
                [_resolve_tool("pytest"), str(test_file), "-v", "--tb=short", "-p", "no:cacheprovider"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workdir),
            )
            passed, failed = _parse_pytest_summary(proc.stdout)
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="pytest",
                exit_code=proc.returncode,
                summary=_truncate(proc.stdout, 2000),
                raw_output_path=str(test_file),
                passed=passed,
                failed=failed,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="pytest",
                exit_code=-1,
                summary="Test execution timed out",
                raw_output_path=str(test_file),
            )
        except FileNotFoundError:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="pytest",
                exit_code=-1,
                summary="pytest not found in PATH",
            )


def run_ruff(
    code: str,
    task_id: str,
    sample_id: int,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
) -> ToolResult:
    with _temp_dir(tmp_dir) as workdir:
        src_file = workdir / f"{_safe_name(task_id)}_{sample_id}.py"
        src_file.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [_resolve_tool("ruff"), "check", str(src_file), "--output-format=text"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workdir),
            )
            warnings = len([l for l in proc.stdout.splitlines() if l.strip()])
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="ruff",
                exit_code=proc.returncode,
                summary=_truncate(proc.stdout, 2000),
                raw_output_path=str(src_file),
                warnings=warnings,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="ruff",
                exit_code=-1,
                summary="Ruff analysis timed out",
            )
        except FileNotFoundError:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="ruff",
                exit_code=-1,
                summary="ruff not found in PATH",
            )


def run_mypy(
    code: str,
    task_id: str,
    sample_id: int,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
) -> ToolResult:
    with _temp_dir(tmp_dir) as workdir:
        src_file = workdir / f"{_safe_name(task_id)}_{sample_id}.py"
        src_file.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [_resolve_tool("mypy"), str(src_file), "--ignore-missing-imports", "--no-error-summary"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workdir),
            )
            warnings = len([l for l in proc.stdout.splitlines() if l.strip()])
            errors = len([l for l in proc.stderr.splitlines() if l.strip()]) if proc.stderr else 0
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="mypy",
                exit_code=proc.returncode,
                summary=_truncate(proc.stdout + "\n" + (proc.stderr or ""), 2000),
                raw_output_path=str(src_file),
                warnings=warnings,
                errors=errors,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="mypy",
                exit_code=-1,
                summary="Mypy analysis timed out",
            )
        except FileNotFoundError:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="mypy",
                exit_code=-1,
                summary="mypy not found in PATH",
            )


def run_bandit(
    code: str,
    task_id: str,
    sample_id: int,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
) -> ToolResult:
    with _temp_dir(tmp_dir) as workdir:
        src_file = workdir / f"{_safe_name(task_id)}_{sample_id}.py"
        src_file.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [_resolve_tool("bandit"), "-r", str(src_file), "-f", "json", "-ll"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(workdir),
            )
            issues = 0
            summary_text = ""
            try:
                data = json.loads(proc.stdout)
                results = data.get("results", [])
                issues = len(results)
                issues_text = "\n".join(
                    f"{r.get('test_id','?')}:{r.get('test_name','?')} line {r.get('line_number','?')}"
                    for r in results[:20]
                )
                summary_text = f"Issues: {issues}\n{issues_text}"
            except json.JSONDecodeError:
                summary_text = proc.stdout[:2000]
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="bandit",
                exit_code=proc.returncode,
                summary=_truncate(summary_text, 2000),
                raw_output_path=str(src_file),
                warnings=issues,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="bandit",
                exit_code=-1,
                summary="Bandit analysis timed out",
            )
        except FileNotFoundError:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="bandit",
                exit_code=-1,
                summary="bandit not found in PATH",
            )


def run_all_tools(
    code: str,
    task: Task,
    sample_id: int,
    tools: Optional[list[str]] = None,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
) -> list[ToolResult]:
    if tools is None:
        tools = ["pytest", "ruff", "mypy", "bandit", "hypothesis"]
    results: list[ToolResult] = []
    for tool in tools:
        if tool == "pytest" and task.test_code:
            results.append(
                run_pytest(code, task.test_code, task.task_id, sample_id, tmp_dir, timeout, entry_point=task.entry_point)
            )
        elif tool == "ruff":
            results.append(run_ruff(code, task.task_id, sample_id, tmp_dir, timeout))
        elif tool == "mypy":
            results.append(run_mypy(code, task.task_id, sample_id, tmp_dir, timeout))
        elif tool == "bandit":
            results.append(run_bandit(code, task.task_id, sample_id, tmp_dir, timeout))
        elif tool == "hypothesis" and task.test_code:
            results.append(
                run_hypothesis(code, task.test_code, task.task_id, sample_id, tmp_dir, timeout, entry_point=task.entry_point)
            )
    return results


def _parse_pytest_summary(output: str) -> tuple[int, int]:
    passed = 0
    failed = 0
    import re

    for line in output.splitlines():
        cleaned = line.lstrip("=").strip()
        if "passed" in cleaned or "failed" in cleaned:
            p = re.search(r"(\d+)\s+passed", cleaned)
            f = re.search(r"(\d+)\s+failed", cleaned)
            if p:
                passed = int(p.group(1))
            if f:
                failed = int(f.group(1))
    return passed, failed


def run_hypothesis(
    code: str,
    test_code: str,
    task_id: str,
    sample_id: int,
    tmp_dir: Optional[Path] = None,
    timeout: int = TIMEOUT_SECONDS,
    entry_point: Optional[str] = None,
) -> ToolResult:
    try:
        import hypothesis
    except ImportError:
        return ToolResult(
            task_id=task_id,
            sample_id=sample_id,
            tool="hypothesis",
            exit_code=-1,
            summary="hypothesis not installed",
        )

    with _temp_dir(tmp_dir) as workdir:
        test_file = workdir / f"htest_{_safe_name(task_id)}_{sample_id}.py"
        check_wrapper = ""
        if entry_point and "def check(" in test_code:
            check_wrapper = f"\n\ndef test_{entry_point}():\n    check({entry_point})\n"
        wrapper = (
            f"{code}\n\n"
            f"from hypothesis import given, strategies as st, settings\n"
            f"from hypothesis import HealthCheck\n\n"
            f"{test_code}\n"
            f"{check_wrapper}\n"
        )
        test_file.write_text(wrapper, encoding="utf-8")

        try:
            proc = subprocess.run(
                [
                    _resolve_tool("pytest"), str(test_file), "-v", "--tb=short",
                    "--hypothesis-show-statistics",
                    "-p", "no:cacheprovider",
                ],
                capture_output=True,
                text=True,
                timeout=timeout * 2,
                cwd=str(workdir),
            )
            passed, failed = _parse_pytest_summary(proc.stdout)
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="hypothesis",
                exit_code=proc.returncode,
                summary=_truncate(proc.stdout, 2000),
                raw_output_path=str(test_file),
                passed=passed,
                failed=failed,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                task_id=task_id,
                sample_id=sample_id,
                tool="hypothesis",
                exit_code=-1,
                summary="Hypothesis testing timed out",
            )


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    suffix = "\n... (truncated)"
    return text[:max_len - len(suffix)] + suffix


from contextlib import contextmanager
import shutil


@contextmanager
def _temp_dir(base: Optional[Path] = None):
    if base:
        base.mkdir(parents=True, exist_ok=True)
        yield base
    else:
        d = Path(tempfile.mkdtemp(prefix="bgebench_"))
        try:
            yield d
        finally:
            shutil.rmtree(d, ignore_errors=True)
