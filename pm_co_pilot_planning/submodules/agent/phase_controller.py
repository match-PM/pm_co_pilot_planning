from dataclasses import dataclass
from typing import Optional

_EXECUTE_KEYWORDS = ["execute", "run the sequence", "run it", "start execution", "run all"]
_FULL_SEQUENCE_KEYWORDS = ["run the sequence", "run all", "start execution", "execute all", "run everything", "execute the sequence", "execute the whole"]


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
            self.full_execution_requested = self._is_full_sequence_request(user_message)
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
        total_actions: int,
        last_executed_index: int = 0,
        sequence_overview: str = "",
    ) -> NextAction:
        """
        Decide what to do after an interaction completes.
        last_executed_index: highest 1-based user index successfully executed this session.
        sequence_overview: structured summary of the whole sequence (names + indices),
            handed over on continuation so the agent sees what the full plan is about.
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
            next_idx = last_executed_index + 1
            if total_actions > 0 and next_idx <= total_actions:
                # Carry the prior summary of key state changes forward. The
                # executor windows its context, so the previous summary message
                # would otherwise fall out of the window and the agent would
                # continue without knowing what has already been assembled.
                handover = ""
                if sequence_overview and sequence_overview.strip():
                    handover += (
                        "Full sequence overview (what the whole plan is about — "
                        "all actions, by 1-based index):\n"
                        f"{sequence_overview.strip()}\n\n"
                    )
                if response_text and response_text.strip():
                    handover += (
                        "State so far — these actions already ran and their effects "
                        "are now part of the scene; do NOT repeat them:\n"
                        f"{response_text.strip()}\n\n"
                    )
                return NextAction(
                    kind="continue",
                    prompt=(
                        f"{handover}"
                        f"Continue executing the sequence from action {next_idx}. "
                        f"There are {total_actions - next_idx + 1} actions remaining "
                        f"(actions {next_idx} to {total_actions})."
                    ),
                )

        return NextAction(kind="done")

    # ── Internal helpers ────────────────────────────────────────────────────

    def _is_full_sequence_request(self, user_message: str) -> bool:
        return any(kw in user_message.lower() for kw in _FULL_SEQUENCE_KEYWORDS)

    def _detect_phase(self, user_message: str) -> str:
        if any(kw in user_message.lower() for kw in _EXECUTE_KEYWORDS):
            return "executing"
        return "planning"
