from bgebench.common.schemas import DefectCategory, ToolResult
from bgebench.common.defect_classifier import (
    classify_defects,
    compute_severity_weighted,
    count_by_category,
    DEFECT_WEIGHTS,
)


def test_classify_pytest_syntax_error():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="pytest",
        exit_code=1,
        summary="SyntaxError: invalid syntax\nE       ^\n",
    )
    defects = classify_defects([tr])
    assert len(defects) > 0
    assert any(d.category == DefectCategory.SYNTAX for d in defects)


def test_classify_pytest_assertion_error():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="pytest",
        exit_code=1,
        summary="AssertionError: assert 1 == 2\nE  assert 1 == 2\n",
    )
    defects = classify_defects([tr])
    assert any(d.category == DefectCategory.SEMANTIC for d in defects)


def test_classify_pytest_name_error():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="pytest",
        exit_code=1,
        summary="NameError: name 'foo' is not defined\n",
    )
    defects = classify_defects([tr])
    assert any(d.category == DefectCategory.RUNTIME for d in defects)


def test_classify_pytest_import_error():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="pytest",
        exit_code=1,
        summary="ModuleNotFoundError: No module named 'numpy'\n",
    )
    defects = classify_defects([tr])
    assert any(d.category == DefectCategory.API for d in defects)


def test_classify_ruff():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="ruff",
        exit_code=1,
        summary="test.py:1:1: F401 'os' imported but unused\n",
    )
    defects = classify_defects([tr])
    assert len(defects) > 0


def test_classify_bandit():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="bandit",
        exit_code=1, summary="B110:try_except_pass\n", warnings=1,
    )
    defects = classify_defects([tr])
    assert any(d.category == DefectCategory.SECURITY for d in defects)


def test_compute_severity_weighted():
    from bgebench.common.schemas import Defect, Severity

    defects = [
        Defect("t", 0, DefectCategory.SYNTAX, "d1", "", Severity.MEDIUM),
        Defect("t", 0, DefectCategory.SECURITY, "d2", "", Severity.CRITICAL),
        Defect("t", 0, DefectCategory.SEMANTIC, "d3", "", Severity.HIGH),
    ]
    weight = compute_severity_weighted(defects)
    expected = DEFECT_WEIGHTS[DefectCategory.SYNTAX] + DEFECT_WEIGHTS[DefectCategory.SECURITY] + DEFECT_WEIGHTS[DefectCategory.SEMANTIC]
    assert weight == expected


def test_count_by_category():
    from bgebench.common.schemas import Defect, Severity

    defects = [
        Defect("t", 0, DefectCategory.SYNTAX, "d1", "", Severity.MEDIUM),
        Defect("t", 0, DefectCategory.SYNTAX, "d2", "", Severity.MEDIUM),
        Defect("t", 0, DefectCategory.SEMANTIC, "d3", "", Severity.HIGH),
    ]
    counts = count_by_category(defects)
    assert counts == {"syntax": 2, "semantic": 1}


def test_empty_tool_results():
    defects = classify_defects([])
    assert len(defects) == 0


def test_ruff_no_issues():
    tr = ToolResult(
        task_id="test", sample_id=0, tool="ruff",
        exit_code=0, summary="", warnings=0,
    )
    defects = classify_defects([tr])
    assert len(defects) == 0
