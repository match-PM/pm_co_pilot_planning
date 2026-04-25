import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from .interaction_logger import InteractionLogger


class LearningManager:
    """
    Post-execution knowledge recording via a single LLM pass.

    After each execution the learner receives the full trace — concrete state
    changes per service, confirmed parameter fixes together with the error
    messages they resolved, and the current KB view — and generates zero or
    more natural-language knowledge entries (general_knowledge or per-service
    usage_notes).  No templated fact tokens are produced.
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

        execution_summary = self._build_execution_summary(interaction_logger)
        state_changes_by_service = execution_summary["state_changes_by_service"]
        fixes_applied = execution_summary["fixes_applied"]
        execution_order = execution_summary["execution_order"]

        exec_interaction_count = sum(
            1 for i in interaction_logger.interactions
            if i.get("phase") in ("executing", "escalated")
        )
        self._service_node.get_logger().info(
            f"Post-execution learning nudge for services: {executed_services} "
            f"(from {exec_interaction_count} execution/escalated interactions)"
        )

        if not execution_order and not fixes_applied:
            self._service_node.get_logger().info(
                "Learning nudge: nothing ran successfully and no fixes — skipping LLM pass."
            )
            interaction_logger.append_raw({
                "timestamp": datetime.now().isoformat(),
                "user_message": "[post-execution learning nudge]",
                "agent_response": "No successful executions to analyze.",
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
                state_changes_by_service=state_changes_by_service,
                execution_order=execution_order,
                fixes_applied=fixes_applied,
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
        state_changes_by_service: dict,
        execution_order: list,
        fixes_applied: list,
    ) -> tuple[str, list, int, int]:
        """Run the natural-language learner over the execution trace."""
        service_names: set = {
            entry.get("service_client") for entry in execution_order
            if entry.get("service_client")
        }
        service_names.update(
            f["service_client"] for f in fixes_applied if f.get("service_client")
        )

        existing_kb: dict = {}
        for svc in sorted(service_names):
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
            "STATE CHANGES observed per service (raw, concrete, deduped):\n"
            f"{json.dumps(state_changes_by_service, indent=2)}\n\n"
            "EXECUTION TRACE (ordered, one entry per action index):\n"
            f"{json.dumps(execution_order, indent=2)}\n\n"
            "CONFIRMED FIXES (parameter changes that resolved failures, with the "
            "error messages they resolved):\n"
            f"{json.dumps(fixes_applied, indent=2)}\n\n"
            "EXISTING KB STATE for involved services (for semantic dedup):\n"
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

    # ── Execution summary ───────────────────────────────────────────────────

    def _build_execution_summary(self, interaction_logger: InteractionLogger) -> dict:
        """Extract state_changes_by_service, confirmed fixes (with failure messages), and ordered trace.

        Fixes are only included when the subsequent re-execution of the same action
        index succeeded.  Each confirmed fix carries the error messages from the
        failed attempt(s) it resolved so the LLM can formulate constraint rules.
        """
        fixes_confirmed: list = []
        state_changes_by_service: dict = {}
        seen_state_change_keys: set = set()  # (service, change) for dedup

        # Build index→(action_name, service_client, parameters) from the live RSAP sequence
        index_to_action: dict = {}
        if self._rsap_instance is not None:
            for i, action in enumerate(self._rsap_instance.action_list):
                user_index = i + 1
                name = ""
                if hasattr(action, "get_name"):
                    name = action.get_name()
                elif hasattr(action, "name"):
                    name = action.name
                client = getattr(action, "client", "")
                params: dict = {}
                if hasattr(action, "get_request_as_ordered_dict"):
                    try:
                        params = dict(action.get_request_as_ordered_dict())
                    except Exception:
                        params = {}
                index_to_action[user_index] = {
                    "action_name": name,
                    "service_client": client,
                    "parameters": params,
                }

        all_steps = interaction_logger.get_execution_steps()

        self._service_node.get_logger().info(
            f"Building execution summary from {len(all_steps)} total execution steps"
        )

        # Pending fixes: index → fix_data, waiting for confirmation by a successful retry
        pending_fixes: dict = {}
        # Failure messages per index for passing to confirmed fixes
        failure_messages_by_index: dict = {}

        # Per-index state: track first-seen order, accumulated state_changes, final success
        index_order: list = []  # indices in the order they are first seen
        outcome_by_index: dict = {}  # index → {"success": bool, "state_changes": [...]}

        def _extract_failure_message(data: dict) -> str:
            """Pull the most informative failure description from a tool result dict."""
            parts = []
            if data.get("message"):
                msg = data["message"]
                if isinstance(msg, dict):
                    parts.append(json.dumps(msg))
                else:
                    parts.append(str(msg))
            if data.get("error_detail"):
                parts.append(str(data["error_detail"]))
            elif data.get("response"):
                parts.append(json.dumps(data["response"]))
            return " | ".join(parts) if parts else json.dumps(data)

        def _handle_tool_result(data: dict) -> None:
            if "success" not in data or "index" not in data:
                return
            idx = data["index"]
            action_meta = index_to_action.get(idx, {})
            svc = action_meta.get("service_client", "")

            # Record first-seen order
            if idx not in outcome_by_index:
                index_order.append(idx)
                outcome_by_index[idx] = {"success": False, "state_changes": []}

            # Accumulate state changes (including partial failures)
            for change in data.get("state_changes", []):
                key = (svc, change)
                if svc and key not in seen_state_change_keys:
                    seen_state_change_keys.add(key)
                    state_changes_by_service.setdefault(svc, []).append(change)
                if change not in outcome_by_index[idx]["state_changes"]:
                    outcome_by_index[idx]["state_changes"].append(change)

            # Capture failure messages before overwriting success flag
            if not data.get("success"):
                msg = _extract_failure_message(data)
                if msg:
                    failure_messages_by_index.setdefault(idx, []).append(msg)

            # Final success wins (retries overwrite)
            if data.get("success"):
                outcome_by_index[idx]["success"] = True

            # Confirm a pending fix when the re-execution succeeds
            if idx in pending_fixes:
                if data.get("success"):
                    fix = pending_fixes.pop(idx)
                    fix["failure_messages"] = failure_messages_by_index.get(idx, [])
                    fixes_confirmed.append(fix)
                # If still failing, keep in pending (another fix attempt may follow)

        for step in all_steps:
            # Track set_action_parameters calls as fix candidates
            for tc in step.get("tool_calls", []):
                if tc["name"] == "set_action_parameters":
                    idx = tc["args"].get("index")
                    if idx is not None:
                        action_meta = index_to_action.get(idx, {})
                        # Overwrite: multiple fixes on same index → keep the last one tried
                        pending_fixes[idx] = {
                            "action_index": idx,
                            "action_name": action_meta.get("action_name", ""),
                            "service_client": action_meta.get("service_client", ""),
                            "parameters_changed": tc["args"].get("parameters", {}),
                        }

            # Single ToolMessage result
            if step.get("type") == "ToolMessage":
                try:
                    data = json.loads(step.get("content", ""))
                    _handle_tool_result(data)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Parallel tool results
            for resp in step.get("tool_responses", []):
                try:
                    data = json.loads(resp.get("content", ""))
                    _handle_tool_result(data)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Build ordered execution trace (successful actions only)
        execution_order: list = []
        for idx in index_order:
            outcome = outcome_by_index[idx]
            if not outcome["success"]:
                continue
            action_meta = index_to_action.get(idx, {})
            execution_order.append({
                "index": idx,
                "service_client": action_meta.get("service_client", ""),
                "parameters": action_meta.get("parameters", {}),
                "state_changes": outcome["state_changes"],
            })

        return {
            "state_changes_by_service": state_changes_by_service,
            "fixes_applied": fixes_confirmed,
            "execution_order": execution_order,
        }
