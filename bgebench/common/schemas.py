from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DefectCategory(str, Enum):
    SYNTAX = "syntax"
    RUNTIME = "runtime"
    SEMANTIC = "semantic"
    BOUNDARY = "boundary"
    SECURITY = "security"
    API = "api"
    MAINTAINABILITY = "maintainability"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Condition(str, Enum):
    BASELINE = "baseline"
    TREATMENT = "treatment"


class Decision(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass
class Task:
    task_id: str
    benchmark: str
    prompt: str
    canonical_solution: Optional[str] = None
    test_code: Optional[str] = None
    entry_point: Optional[str] = None


@dataclass
class Generation:
    task_id: str
    benchmark: str
    model: str
    prompt_variant: str
    sample_id: int
    generated_code: str
    generated_loc: int
    generated_tokens: int
    accepted_loc: int = 0
    generation_time_s: float = 0.0
    condition: Condition = Condition.BASELINE


@dataclass
class ToolResult:
    task_id: str
    sample_id: int
    tool: str
    exit_code: int
    summary: str
    raw_output_path: str = ""
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    errors: int = 0


@dataclass
class Defect:
    task_id: str
    sample_id: int
    category: DefectCategory
    description: str
    location: str = ""
    severity: Severity = Severity.MEDIUM
    tool_source: str = ""


@dataclass
class FeedbackItem:
    defect_type: str
    location: str
    evidence: str
    severity: str
    repair_instruction: str


@dataclass
class RepairIteration:
    task_id: str
    sample_id: int
    iteration: int
    code: str
    loc: int
    pytest_failed: int
    ruff_warnings: int
    mypy_warnings: int
    bandit_warnings: int
    severity_weighted_defects: float
    hard_pass: bool
    wall_time_s: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ReliabilityResult:
    task_id: str
    benchmark: str
    model: str
    prompt_variant: str
    sample_id: int
    generated_loc: int
    generated_tokens: int
    defects_raw: int
    defects_weighted: float
    repair_iterations: int
    final_failure: bool
    pass_rate_public: float
    pass_rate_hidden: float = -1.0
    rrs: float = 0.0
    decision: Decision = Decision.REVIEW
    decision_threshold_version: str = "default"
