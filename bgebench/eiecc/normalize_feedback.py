import logging

from bgebench.common.schemas import Defect, FeedbackItem, ToolResult

logger = logging.getLogger(__name__)


def normalize_feedback(
    tool_results: list[ToolResult],
    defects: list[Defect],
) -> list[FeedbackItem]:
    items: list[FeedbackItem] = []
    for defect in defects:
        tool_output = ""
        for tr in tool_results:
            if tr.tool == defect.tool_source and tr.task_id == defect.task_id and tr.sample_id == defect.sample_id:
                tool_output = tr.summary[:500]
                break

        item = FeedbackItem(
            defect_type=defect.category.value,
            location=defect.location or "unknown",
            evidence=tool_output[:200] if tool_output else defect.description,
            severity=defect.severity.value,
            repair_instruction=_build_repair_instruction(defect, tool_output),
        )
        items.append(item)
    return items


def _build_repair_instruction(defect: Defect, evidence: str) -> str:
    category = defect.category.value
    if category == "syntax":
        return (
            f"Fix the syntax error: {defect.description}. "
            f"Ensure the code can be parsed and executed by Python."
        )
    elif category == "runtime":
        return (
            f"Fix the runtime error: {defect.description}. "
            f"Ensure the function handles valid inputs without raising exceptions."
        )
    elif category == "semantic":
        return (
            f"Fix the semantic error: {defect.description}. "
            f"Ensure the function returns the correct output for all valid inputs."
        )
    elif category == "boundary":
        return (
            f"Fix the boundary-condition bug. {defect.description}. "
            f"Test edge cases: empty input, None, extreme values."
        )
    elif category == "security":
        return (
            f"Fix the security issue: {defect.description}. "
            f"Remove or replace unsafe patterns. Prefer safe alternatives."
        )
    elif category == "api":
        return (
            f"Fix the API usage error: {defect.description}. "
            f"Verify the function, module, or class name exists in Python."
        )
    elif category == "maintainability":
        return (
            f"Fix the code style issue: {defect.description}. "
            f"Follow Python style conventions (PEP 8)."
        )
    return f"Fix the following issue: {defect.description}"


def build_repair_prompt(
    original_code: str,
    feedback_items: list[FeedbackItem],
) -> str:
    feedback_text = "\n".join(
        f"- [{item.severity}] {item.defect_type}: {item.repair_instruction} "
        f"(location: {item.location}, evidence: {item.evidence[:100]})"
        for item in feedback_items
    )

    return (
        f"The following Python code has verification issues.\n\n"
        f"```python\n{original_code}\n```\n\n"
        f"Issues found:\n{feedback_text}\n\n"
        f"Please fix the code to address all the issues listed above. "
        f"Preserve existing correct behavior. "
        f"Make minimal changes — fix only what is broken. "
        f"Return only the corrected Python code, no explanation."
    )
