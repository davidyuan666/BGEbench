from bgebench.common.verification import _parse_pytest_summary, _safe_name, _truncate


def test_parse_pytest_summary_all_pass():
    output = "tests/test_foo.py::test_add PASSED\n====== 4 passed in 0.05s ======"
    passed, failed = _parse_pytest_summary(output)
    assert passed == 4
    assert failed == 0


def test_parse_pytest_summary_mixed():
    output = "tests/test_foo.py::test_add PASSED\ntests/test_foo.py::test_sub FAILED\n===== 1 passed, 1 failed in 0.05s ====="
    passed, failed = _parse_pytest_summary(output)
    assert passed == 1
    assert failed == 1


def test_parse_pytest_summary_unknown():
    output = "something went wrong"
    passed, failed = _parse_pytest_summary(output)
    assert passed == 0
    assert failed == 0


def test_safe_name():
    assert _safe_name("HumanEval/0") == "HumanEval_0"
    assert _safe_name("test task") == "test_task"


def test_truncate():
    assert _truncate("hello", 100) == "hello"
    assert len(_truncate("a" * 3000, 100)) <= 104
