import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .interaction_logger import InteractionLogger


class LearningManager:
    """
    Post-execution knowledge recording via a single LLM pass.

    After each execution the learner receives the raw, ordered execution
    transcript (every executor tool call and result, including failures,
    retries, and escalations) together with the current KB view, and generates
    zero or more natural-language knowledge entries (general_knowledge or
    per-service usage_notes).  Nothing is pre-extracted, so no failure or
    recovery is filtered out before the learner sees it.  No templated fact
    tokens are produced.
    """

    def __init__(self, learning_model, learning_tools, knowledge_tools, service_node,
                 system_prompt: str, rsap_instance=None):
        self._learning_model = learning_model
        self._learning_tools = learning_tools
        self._knowledge_tools = knowledge_tools
        self._service_node = service_node
        self._system_prompt = system_prompt
        self._rsap_instance = rsap_instance

    def run(
        self,
        interaction_logger: InteractionLogger,
        executed_services: list,
        learner_model_name: str,
    ):
        """Single LLM pass that generates natural-language knowledge entries."""
        if not executed_services:
            self._service_node.get_logger().info(
                "Learning nudge skipped: no service clients found in sequence."
            )
            return

        transcript = self._collect_transcript(interaction_logger)

        exec_interaction_count = sum(
            1 for i in interaction_logger.interactions
            if i.get("phase") in ("executing", "escalated")
        )
        self._service_node.get_logger().info(
            f"Post-execution learning nudge for services: {executed_services} "
            f"(from {exec_interaction_count} execution/escalated interactions, "
            f"{len(transcript)} transcript steps)"
        )

        if not transcript:
            self._service_node.get_logger().info(
                "Learning nudge: empty execution transcript — skipping LLM pass."
            )
            interaction_logger.append_raw({
                "timestamp": datetime.now().isoformat(),
                "user_message": "[post-execution learning nudge]",
                "agent_response": "No execution transcript to analyze.",
                "phase": "learning",
                "model": learner_model_name,
                "steps": 0,
                "step_details": [],
                "tokens": {"input": 0, "output": 0, "total": 0},
            })
            return

        learning_start = datetime.now()
        try:
            result_text, step_details, input_tokens, output_tokens = self._run_learning_agent(
                transcript=transcript,
                service_names=executed_services,
            )
        except Exception as e:
            self._service_node.get_logger().warning(
                f"Post-execution learning LLM pass failed (non-critical): {e}"
            )
            return

        learning_end = datetime.now()
        self._service_node.get_logger().info(
            f"Learning nudge completed: {result_text[:200]}"
        )

        interaction_logger.append_raw({
            "timestamp": learning_start.isoformat(),
            "timestamp_end": learning_end.isoformat(),
            "execution_time_seconds": (learning_end - learning_start).total_seconds(),
            "user_message": "[post-execution learning nudge]",
            "agent_response": result_text,
            "phase": "learning",
            "model": learner_model_name,
            "steps": len(step_details),
            "step_details": step_details,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
        })

    # ── LangGraph hook ──────────────────────────────────────────────────────

    def _pre_model_hook(self, state):
        messages = state["messages"]
        return {
            "llm_input_messages": [SystemMessage(content=self._system_prompt)] + list(messages)
        }

    # ── LLM pass ────────────────────────────────────────────────────────────

    def _run_learning_agent(
        self,
        transcript: list,
        service_names: list,
    ) -> tuple[str, list, int, int]:
        """Run the natural-language learner over the raw execution transcript."""
        existing_kb: dict = {}
        for svc in sorted(set(service_names)):
            raw = self._knowledge_tools.get_knowledge_for_services([svc])
            data = json.loads(raw)
            entry = data.get("services", {}).get(svc, {})
            # Expose id+note pairs so the learner can confirm/contradict by id.
            # get_knowledge_for_services already returns {"id": ..., "note": ...} dicts.
            existing_kb[svc] = {
                "usage_notes": [
                    n for n in entry.get("usage_notes", [])
                    if isinstance(n, dict) and n.get("note")
                ],
            }

        human_message = (
            "EXECUTION TRANSCRIPT (raw, ordered — every executor tool call and "
            "result, including failures, retries, and escalations; each result "
            "embeds its own success flag, error message, and state_changes):\n"
            f"{json.dumps(transcript, indent=2)}\n\n"
            "EXISTING KB STATE for involved services (for classification and "
            "semantic dedup):\n"
            f"{json.dumps(existing_kb, indent=2)}"
        )

        agent_executor = create_react_agent(
            model=self._learning_model,
            tools=self._learning_tools,
            pre_model_hook=self._pre_model_hook,
        )

        step_details = []
        step_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_logged_message = None
        result_text = ""

        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=human_message)]},
            {"recursion_limit": 60},
            stream_mode="values",
        ):
            last_message = step["messages"][-1]
            if last_logged_message is not None and last_message is last_logged_message:
                continue

            step_count += 1
            last_logged_message = last_message

            step_log = {
                "step": step_count,
                "type": type(last_message).__name__,
                "phase": "learning",
                "content": str(last_message.content) if hasattr(last_message, "content") else None,
            }

            if hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
                usage = last_message.usage_metadata
                if isinstance(usage, dict):
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                else:
                    total_input_tokens += getattr(usage, "input_tokens", 0)
                    total_output_tokens += getattr(usage, "output_tokens", 0)

            self._service_node.get_logger().info(
                f"Learning step {step_count}: {type(last_message).__name__} - "
                f"{str(last_message.content)[:200]}"
            )

            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_calls_log = []
                for tc in last_message.tool_calls:
                    info = {"name": tc.get("name", "unknown"), "args": tc.get("args", {})}
                    tool_calls_log.append(info)
                    self._service_node.get_logger().info(
                        f"  → Tool call: {info['name']} with args: {info['args']}"
                    )
                step_log["tool_calls"] = tool_calls_log

            if isinstance(last_message, ToolMessage):
                tool_responses = []
                for msg in reversed(step["messages"]):
                    if isinstance(msg, ToolMessage):
                        tool_responses.insert(0, {
                            "tool_call_id": getattr(msg, "tool_call_id", "unknown"),
                            "name": getattr(msg, "name", "unknown"),
                            "content": str(msg.content),
                        })
                    elif not isinstance(msg, ToolMessage):
                        break
                if len(tool_responses) > 1:
                    step_log["tool_responses"] = tool_responses
                    self._service_node.get_logger().info(
                        f"  → Received {len(tool_responses)} tool responses"
                    )
                    for i, resp in enumerate(tool_responses, 1):
                        self._service_node.get_logger().info(f"     {i}. {resp['content']}")

            step_details.append(step_log)

            if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
                result_text = str(last_message.content)
                self._service_node.get_logger().info(
                    f"Learning LLM pass completed after {step_count} steps — "
                    f"tokens in/out: {total_input_tokens}/{total_output_tokens}"
                )
                break

        return result_text, step_details, total_input_tokens, total_output_tokens

    # ── Transcript collection ───────────────────────────────────────────────

    def _collect_transcript(self, interaction_logger: InteractionLogger) -> list:
        """Return the raw, ordered execution transcript for the learner.

        Forwards the executor's actual tool calls and results verbatim (every
        result already embeds its success flag, error message, and state_changes)
        so the learner sees failures, retries, and escalations losslessly and
        decides for itself what is worth recording. Steps whose only content is an
        empty assistant message are dropped to keep the payload lean.
        """
        all_steps = interaction_logger.get_execution_steps()

        transcript: list = []
        for step in all_steps:
            content = step.get("content")
            tool_calls = step.get("tool_calls")
            tool_responses = step.get("tool_responses")

            # Skip empty assistant turns (no content, no tool calls) — pure noise.
            if not content and not tool_calls and not tool_responses:
                continue

            entry: dict = {"type": step.get("type")}
            if content:
                entry["content"] = content
            if tool_calls:
                entry["tool_calls"] = tool_calls
            if tool_responses:
                entry["tool_responses"] = tool_responses
            transcript.append(entry)

        self._service_node.get_logger().info(
            f"Collected execution transcript: {len(transcript)} steps "
            f"(from {len(all_steps)} raw execution steps)"
        )
        return transcript
