import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .interaction_logger import InteractionLogger


class LearningManager:
    """
    Post-execution knowledge recording.
    Runs a stateless streaming learning agent after any session that included execution
    or escalation — fixing the original bug where escalated sessions were skipped.
    """

    def __init__(self, learning_model, learning_tools, knowledge_tools, service_node, system_prompt: str):
        self._learning_model = learning_model
        self._learning_tools = learning_tools
        self._knowledge_tools = knowledge_tools
        self._service_node = service_node
        self._system_prompt = system_prompt

    def run(
        self,
        interaction_logger: InteractionLogger,
        executed_services: list,
        learner_model_name: str,
    ):
        """Build execution summary, run streaming learning agent, append result to logger."""
        if not executed_services:
            self._service_node.get_logger().info(
                "Learning nudge skipped: no service clients found in sequence."
            )
            return

        targeted_kb = self._knowledge_tools.get_knowledge_for_services(executed_services)
        execution_summary = self._build_execution_summary(interaction_logger)

        exec_interaction_count = sum(
            1 for i in interaction_logger.interactions
            if i.get("phase") in ("executing", "escalated")
        )
        self._service_node.get_logger().info(
            f"Post-execution learning nudge for services: {executed_services} "
            f"(summary from {exec_interaction_count} execution/escalated interactions)"
        )

        learning_start = datetime.now()
        try:
            result_text, step_details, input_tokens, output_tokens = self._run_learning_agent(
                targeted_kb, execution_summary
            )
        except Exception as e:
            self._service_node.get_logger().warning(
                f"Post-execution learning call failed (non-critical): {e}"
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

    # ── Streaming agent ─────────────────────────────────────────────────────

    def _run_learning_agent(
        self, targeted_kb: str, execution_summary: str
    ) -> tuple[str, list, int, int]:
        agent_executor = create_react_agent(
            model=self._learning_model,
            tools=self._learning_tools,
            pre_model_hook=self._pre_model_hook,
            # No checkpointer — fully stateless, isolated from main agent memory
        )

        human_message = (
            f"Current knowledge base state:\n{targeted_kb}\n\n"
            f"Execution summary (what actually happened during this run):\n{execution_summary}\n\n"
            "Instructions:\n"
            "- Use the execution summary to understand what services ran, what failed, "
            "what fixes were applied, and what state changes occurred.\n"
            "- For services with 'not_in_kb: true': call get_service_parameters first, "
            "then record each parameter.\n"
            "- For failed actions: record the error cause as a usage_note on that service.\n"
            "- For successful fixes (fixes_applied): record the fix as a usage_note.\n"
            "- For state_changes from successful actions: use them to infer postconditions.\n"
            "- For services with empty preconditions: infer from the execution order "
            "which services had to succeed first.\n"
            "- For actions that appear unnecessary or redundant (e.g., a fix was applied to the same "
            "parameter on multiple actions of the same service type): record an optimization note as a "
            "usage_note on the relevant service so future plans avoid the mistake."
        )

        step_details = []
        step_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        last_logged_message = None
        result_text = ""

        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=human_message)]},
            {"recursion_limit": 80},
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
                    f"Learning agent completed after {step_count} steps — "
                    f"tokens in/out: {total_input_tokens}/{total_output_tokens}"
                )
                break

        return result_text, step_details, total_input_tokens, total_output_tokens

    # ── Execution summary ───────────────────────────────────────────────────

    def _build_execution_summary(self, interaction_logger: InteractionLogger) -> str:
        """Extract a compact execution summary from all execution/escalated interactions."""
        action_results = []
        fixes_applied = []
        knowledge_recorded = []
        seen_action_keys = set()

        all_steps = interaction_logger.get_execution_steps()

        self._service_node.get_logger().info(
            "Building execution summary from %d total execution steps",
            len(all_steps),
        )

        for step in all_steps:
            for tc in step.get("tool_calls", []):
                if tc["name"] == "set_action_parameters":
                    fixes_applied.append({
                        "action_index": tc["args"].get("index"),
                        "parameters_changed": tc["args"].get("parameters", {}),
                    })
                elif tc["name"] == "record_knowledge":
                    knowledge_recorded.append({
                        "service": tc["args"].get("service_name", ""),
                        "field": tc["args"].get("field", ""),
                        "content": tc["args"].get("content", ""),
                    })

            def _parse(data):
                idx = data.get("index")
                key = (idx, data.get("success"))
                if key in seen_action_keys:
                    return None
                seen_action_keys.add(key)
                entry = {
                    "index": idx,
                    "action_name": data.get("action_name"),
                    "success": data.get("success"),
                    "service_client": data.get("service_client", ""),
                }
                if not data.get("success"):
                    response = data.get("response", {})
                    entry["error"] = (
                        response.get("message") if isinstance(response, dict)
                        else data.get("message", "")
                    )
                if data.get("state_changes"):
                    entry["state_changes"] = data["state_changes"]
                return entry

            if step.get("type") == "ToolMessage":
                try:
                    data = json.loads(step.get("content", ""))
                    if "action_name" in data and "success" in data:
                        entry = _parse(data)
                        if entry:
                            action_results.append(entry)
                except (json.JSONDecodeError, TypeError):
                    pass

            for resp in step.get("tool_responses", []):
                try:
                    data = json.loads(resp.get("content", ""))
                    if "action_name" in data and "success" in data:
                        entry = _parse(data)
                        if entry:
                            action_results.append(entry)
                except (json.JSONDecodeError, TypeError):
                    pass

        return json.dumps(
            {
                "actions_executed": action_results,
                "fixes_applied": fixes_applied,
                "knowledge_already_recorded": knowledge_recorded,
            },
            indent=2,
        )
