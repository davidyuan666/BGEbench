import re
import logging
from typing import Optional

from bgebench.common.schemas import Defect, DefectCategory, Severity, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_SEVERITY: dict[DefectCategory, Severity] = {
    DefectCategory.SYNTAX: Severity.MEDIUM,
    DefectCategory.RUNTIME: Severity.HIGH,
    DefectCategory.SEMANTIC: Severity.HIGH,
    DefectCategory.BOUNDARY: Severity.HIGH,
    DefectCategory.SECURITY: Severity.CRITICAL,
    DefectCategory.API: Severity.MEDIUM,
    DefectCategory.MAINTAINABILITY: Severity.LOW,
}

DEFECT_WEIGHTS: dict[DefectCategory, int] = {
    DefectCategory.SYNTAX: 1,
    DefectCategory.RUNTIME: 2,
    DefectCategory.SEMANTIC: 3,
    DefectCategory.BOUNDARY: 3,
    DefectCategory.SECURITY: 4,
    DefectCategory.API: 2,
    DefectCategory.MAINTAINABILITY: 1,
}


def classify_defects(tool_results: list[ToolResult]) -> list[Defect]:
    defects: list[Defect] = []
    for tr in tool_results:
        if tr.tool == "pytest":
            defects.extend(_classify_pytest(tr))
        elif tr.tool == "ruff":
            defects.extend(_classify_ruff(tr))
        elif tr.tool == "mypy":
            defects.extend(_classify_mypy(tr))
        elif tr.tool == "bandit":
            defects.extend(_classify_bandit(tr))
    return defects


def compute_severity_weighted(defects: list[Defect]) -> float:
    return sum(DEFECT_WEIGHTS.get(d.category, 1) for d in defects)


def count_by_category(defects: list[Defect]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in defects:
        key = d.category.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def _classify_pytest(tr: ToolResult) -> list[Defect]:
    defects: list[Defect] = []
    output = tr.summary
    if tr.exit_code == 0:
        return defects

    assertion_pattern = re.compile(r"AssertionError:\s*(.*)")
    name_error_pattern = re.compile(r"NameError:\s*name\s+'(\w+)'")
    import_error_pattern = re.compile(r"ImportError:\s*|ModuleNotFoundError:\s*")
    syntax_error_pattern = re.compile(r"SyntaxError:\s*(.*)")
    type_error_pattern = re.compile(r"TypeError:\s*(.*)")
    index_error_pattern = re.compile(r"IndexError:\s*(.*)")
    key_error_pattern = re.compile(r"KeyError:\s*(.*)")
    value_error_pattern = re.compile(r"ValueError:\s*(.*)")
    zero_division_pattern = re.compile(r"ZeroDivisionError:\s*(.*)")
    overflow_pattern = re.compile(r"OverflowError:\s*(.*)")
    recursion_pattern = re.compile(r"RecursionError:\s*(.*)")

    for match in syntax_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.SYNTAX,
                description=f"Syntax error: {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.SYNTAX],
                tool_source="pytest",
            )
        )
    for match in assertion_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.SEMANTIC,
                description=f"Assertion failed: {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.SEMANTIC],
                tool_source="pytest",
            )
        )
    for match in name_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.RUNTIME,
                description=f"NameError: '{match.group(1)}' not defined",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.RUNTIME],
                tool_source="pytest",
            )
        )
    if import_error_pattern.search(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.API,
                description="Import/module not found",
                severity=DEFAULT_SEVERITY[DefectCategory.API],
                tool_source="pytest",
            )
        )
    for match in type_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.RUNTIME,
                description=f"TypeError: {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.RUNTIME],
                tool_source="pytest",
            )
        )

    for match in index_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"IndexError (out-of-bounds): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )
    for match in key_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"KeyError (missing key): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )
    for match in value_error_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"ValueError (invalid input): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )
    for match in zero_division_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"ZeroDivisionError (edge case): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )
    for match in overflow_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"OverflowError (numeric boundary): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )
    for match in recursion_pattern.finditer(output):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.BOUNDARY,
                description=f"RecursionError (stack boundary): {match.group(1)[:200]}",
                location=_find_line(output, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.BOUNDARY],
                tool_source="pytest",
            )
        )

    if not defects and tr.exit_code != 0 and tr.failed > 0:
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.RUNTIME,
                description=f"Test failures: {tr.failed} tests failed",
                severity=Severity.MEDIUM,
                tool_source="pytest",
            )
        )
    return defects


def _classify_ruff(tr: ToolResult) -> list[Defect]:
    defects: list[Defect] = []
    if not tr.summary.strip() or tr.exit_code == 0:
        return defects
    rule_pattern = re.compile(r"([A-Z]+\d+)\s+(.*)")
    for line in tr.summary.splitlines():
        match = rule_pattern.search(line)
        if match:
            rule, desc = match.group(1), match.group(2)
            if rule.startswith("E") or rule.startswith("F"):
                category = DefectCategory.SYNTAX
            elif rule.startswith("S"):
                category = DefectCategory.SECURITY
            else:
                category = DefectCategory.MAINTAINABILITY
            defects.append(
                Defect(
                    task_id=tr.task_id,
                    sample_id=tr.sample_id,
                    category=category,
                    description=f"ruff {rule}: {desc[:200]}",
                    location=_find_line(line, 0),
                    severity=DEFAULT_SEVERITY[category],
                    tool_source="ruff",
                )
            )
    if not defects:
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.MAINTAINABILITY,
                description=f"ruff warnings: {tr.warnings} issues",
                severity=Severity.LOW,
                tool_source="ruff",
            )
        )
    return defects


def _classify_mypy(tr: ToolResult) -> list[Defect]:
    defects: list[Defect] = []
    if not tr.summary.strip() or (tr.exit_code == 0 and tr.warnings == 0):
        return defects
    error_pattern = re.compile(r"error:\s*(.*)")
    for match in error_pattern.finditer(tr.summary):
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.API,
                description=f"Type error: {match.group(1)[:200]}",
                location=_find_line(tr.summary, match.start()),
                severity=DEFAULT_SEVERITY[DefectCategory.API],
                tool_source="mypy",
            )
        )
    if not defects and tr.warnings > 0:
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.MAINTAINABILITY,
                description=f"mypy warnings: {tr.warnings} issues",
                severity=Severity.LOW,
                tool_source="mypy",
            )
        )
    return defects


def _classify_bandit(tr: ToolResult) -> list[Defect]:
    defects: list[Defect] = []
    if tr.warnings == 0:
        return defects
    for line in tr.summary.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Issues: 0") or stripped == "Issues: 0":
            continue
        defects.append(
            Defect(
                task_id=tr.task_id,
                sample_id=tr.sample_id,
                category=DefectCategory.SECURITY,
                description=f"Security issue: {stripped[:200]}",
                location=_find_line(line, 0),
                severity=DEFAULT_SEVERITY[DefectCategory.SECURITY],
                tool_source="bandit",
            )
        )
    return defects


def _find_line(text: str, pos: int) -> str:
    before = text[:pos]
    line_num = before.count("\n") + 1
    return f"line ~{line_num}"
