from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class InteractionContext:
    """Mutable context accumulating data for one LLM interaction."""
    user_message: str
    phase: str
    model: str
    start: datetime = field(default_factory=datetime.now)
    step_details: List[dict] = field(default_factory=list)
    step_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class InteractionLogger:
    """
    Owns the full interaction history for the session.
    Provides aggregated access to step_details across all execution-phase interactions.
    """

    def __init__(self):
        self.interactions: list = []

    def start(self, user_message: str, phase: str, model: str) -> InteractionContext:
        return InteractionContext(user_message=user_message, phase=phase, model=model)

    def finish(self, ctx: InteractionContext, response_text: str) -> dict:
        end = datetime.now()
        record = {
            "timestamp": ctx.start.isoformat(),
            "timestamp_end": end.isoformat(),
            "execution_time_seconds": (end - ctx.start).total_seconds(),
            "user_message": ctx.user_message,
            "agent_response": response_text,
            "phase": ctx.phase,
            "model": ctx.model,
            "steps": ctx.step_count,
            "step_details": ctx.step_details,
            "tokens": {
                "input": ctx.total_input_tokens,
                "output": ctx.total_output_tokens,
                "total": ctx.total_input_tokens + ctx.total_output_tokens,
            },
        }
        self.interactions.append(record)
        return record

    def append_raw(self, record: dict):
        """Append a pre-built record (e.g. the learning-phase entry)."""
        self.interactions.append(record)

    def get_execution_steps(self) -> list:
        """Return all step_details from executing/escalated interactions, in order."""
        steps = []
        for interaction in self.interactions:
            if interaction.get("phase") in ("executing", "escalated"):
                steps.extend(interaction.get("step_details", []))
        return steps

    def had_execution(self) -> bool:
        return any(
            i.get("phase") in ("executing", "escalated")
            for i in self.interactions
        )
