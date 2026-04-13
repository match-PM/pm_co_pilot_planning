import json
from dataclasses import dataclass
from typing import Optional

_EXECUTE_KEYWORDS = ["execute", "run the sequence", "run it", "start execution", "run all"]


@dataclass
class NextAction:
    kind: str           # "escalate" | "continue" | "done"
    prompt: Optional[str] = None


class PhaseController:
    """
    Pure state machine — owns phase, failure counter, and continuation intent.
    No LLM calls, no I/O. All decisions are deterministic given inputs.
    """

    def __init__(self):
        self.current_phase: str = "planning"
        self.consecutive_exec_failures: int = 0
        self.full_execution_requested: bool = False

    # ── Phase transitions ───────────────────────────────────────────────────

    def set_from_message(self, user_message: str):
        """Set phase from an initial user message (not a continuation)."""
        new_phase = self._detect_phase(user_message)
        if new_phase != "executing" and self.current_phase == "executing":
            self.consecutive_exec_failures = 0
        if new_phase == "executing" and self.current_phase != "executing":
            self.full_execution_requested = True
        self.current_phase = new_phase

    def mark_escalated(self):
        self.consecutive_exec_failures = 0
        self.current_phase = "escalated"

    def reset_to_planning(self):
        self.current_phase = "planning"
        self.full_execution_requested = False

    # ── Failure tracking ────────────────────────────────────────────────────

    def record_exec_result(self, success: bool):
        """Called by ExecutionMonitor for each execute_single_action response."""
        if success:
            self.consecutive_exec_failures = 0
        else:
            self.consecutive_exec_failures += 1

    def should_escalate(self, response_text: str) -> bool:
        return (
            self.current_phase == "executing"
            and ("ESCALATE:" in response_text or self.consecutive_exec_failures >= 3)
        )

    # ── Decision after one LLM interaction ─────────────────────────────────

    def decide_next(
        self,
        response_text: str,
        step_details: list,
        total_actions: int,
    ) -> NextAction:
        """
        Decide what to do after an interaction completes.
        Returns NextAction with kind "escalate", "continue", or "done".
        """
        # Escalation takes priority over continuation
        if self.should_escalate(response_text):
            return NextAction(
                kind="escalate",
                prompt=(
                    "The execution monitor could not resolve this error and escalated to you.\n"
                    f"Recent response: {response_text}\n"
                    "Please diagnose the root cause using query_assembly_knowledge and fix the sequence. "
                    "After fixing, continue execution from the failed action."
                ),
            )

        # Auto-continuation: executor stopped mid-sequence
        if self.current_phase == "executing" and self.full_execution_requested:
            last_executed = _find_last_executed_index(step_details)
            next_idx = last_executed + 1
            if total_actions > 0 and next_idx <= total_actions:
                return NextAction(
                    kind="continue",
                    prompt=(
                        f"Continue executing the sequence from action {next_idx}. "
                        f"There are {total_actions - next_idx + 1} actions remaining "
                        f"(actions {next_idx} to {total_actions})."
                    ),
                )

        return NextAction(kind="done")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _detect_phase(self, user_message: str) -> str:
        if any(kw in user_message.lower() for kw in _EXECUTE_KEYWORDS):
            return "executing"
        return "planning"


def _find_last_executed_index(step_details: list) -> int:
    """Return the highest successfully executed action index from step_details."""
    last = 0
    for step in step_details:
        if step.get("type") == "ToolMessage":
            try:
                d = json.loads(step.get("content", ""))
                if d.get("success") and "index" in d and "action_name" in d:
                    last = max(last, int(d["index"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        for resp in step.get("tool_responses", []):
            try:
                d = json.loads(resp.get("content", ""))
                if d.get("success") and "index" in d and "action_name" in d:
                    last = max(last, int(d["index"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
    return last
