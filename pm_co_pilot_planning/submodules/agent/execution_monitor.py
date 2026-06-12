import json
from typing import Callable

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .interaction_logger import InteractionContext


class ExecutionResult:
    def __init__(self, response_text: str, interrupted: bool = False):
        self.response_text = response_text
        self.interrupted = interrupted


class ExecutionMonitor:
    """
    Runs ONE LLM invocation: user message → streamed steps → final response.
    Populates an InteractionContext with step_details and token counts.
    Has no control-flow logic — escalation and continuation decisions belong
    to PhaseController and AgentOrchestrator.
    """

    def __init__(
        self,
        assembly_knowledge,
        planner_system_prompt: str,
        executor_system_prompt: str,
        executor_context_window: int,
        planner_model,
        executor_model,
        tools: list,
        memory: MemorySaver,
        langgraph_config: dict,
        service_node,
    ):
        self._assembly_knowledge = assembly_knowledge
        self._planner_system_prompt = planner_system_prompt
        self._executor_system_prompt = executor_system_prompt
        self._executor_context_window = executor_context_window
        self._planner_model = planner_model
        self._executor_model = executor_model
        self._tools = tools
        self._memory = memory
        self._config = langgraph_config
        self._service_node = service_node
        # Set before each run_once call by the orchestrator
        self._current_phase: str = "planning"
        self._session_goal: str = ""

    # ── LangGraph hooks ─────────────────────────────────────────────────────

    def _select_model(self, state, runtime=None):
        if self._current_phase == "executing":
            return self._executor_model
        return self._planner_model

    # Fields kept when slimming a successful execute-action result for the LLM.
    _SLIM_SUCCESS_KEYS = ("success", "index", "start_index", "final_index", "state_changes")

    def _slim_tool_result(self, msg):
        """Return a token-light copy of a SUCCESSFUL execute-action tool result
        for the executor's view: keep success/index/state_changes, drop the
        verbose service `message` payload. Failures and all other messages pass
        through unchanged so the full error reaches the model.

        Only the LLM-facing copy is slimmed — the original (stored) message is
        never mutated, so the interaction log and learner transcript stay full.
        """
        if not isinstance(msg, ToolMessage):
            return msg
        if getattr(msg, "name", "") not in ("execute_single_action", "execute_sequence"):
            return msg
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            return msg
        if not isinstance(data, dict) or not data.get("success"):
            return msg  # keep full content for failures (and unparseable shapes)

        slim = {k: data[k] for k in self._SLIM_SUCCESS_KEYS if k in data}
        return ToolMessage(
            content=json.dumps(slim),
            tool_call_id=msg.tool_call_id,
            name=getattr(msg, "name", None),
        )

    def _pre_model_hook(self, state):
        """Inject system prompt + scene summary; window context for executor."""
        messages = state["messages"]
        scene_msg = SystemMessage(content=self._assembly_knowledge.get_compact_scene_summary())

        if self._current_phase != "executing":
            return {
                "llm_input_messages": [
                    SystemMessage(content=self._planner_system_prompt),
                    scene_msg,
                ] + list(messages)
            }

        # Execution: window to recent messages only
        first_human = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                first_human = msg
                break

        recent = list(messages[-self._executor_context_window:])
        # Drop orphaned ToolMessages at the window boundary
        while recent and isinstance(recent[0], ToolMessage):
            recent.pop(0)

        windowed = [SystemMessage(content=self._executor_system_prompt), scene_msg]
        # Pin the session goal as a fresh SystemMessage every turn so it is always
        # present regardless of windowing — it otherwise rides only inside a single
        # un-pinned handover HumanMessage and scrolls out of the window mid-run.
        if self._session_goal and self._session_goal.strip():
            windowed.append(SystemMessage(
                content=f"ASSEMBLY GOAL (keep in view): {self._session_goal.strip()}"
            ))
        if first_human and first_human not in recent:
            windowed.append(first_human)
        windowed.extend(self._slim_tool_result(m) for m in recent)

        # Only dump the handover when the window actually trims history — that is
        # the only case where the executor loses messages and the handover is
        # worth inspecting. Log each message in full, skipping the static
        # executor system prompt.
        if len(messages) > self._executor_context_window:
            handover_dump = "\n".join(
                f"  [{i}] {type(m).__name__}: {str(m.content)}"
                for i, m in enumerate(windowed)
                if m.content != self._executor_system_prompt
            )
            self._service_node.get_logger().info(
                f"Executor handover — window trimmed "
                f"({len(messages)} → {len(windowed)} messages):\n{handover_dump}"
            )

        return {"llm_input_messages": windowed}

    # ── Main entry point ────────────────────────────────────────────────────

    def run_once(
        self,
        user_message: str,
        phase: str,
        ctx: InteractionContext,
        phase_controller,
        stop_predicate: Callable[[], bool],
        session_goal: str = "",
    ) -> ExecutionResult:
        """
        Stream one agent invocation. Writes all step data into ctx.
        Returns ExecutionResult with the final response text.
        """
        self._current_phase = phase
        self._session_goal = session_goal

        agent_executor = create_react_agent(
            model=self._select_model,
            tools=self._tools,
            checkpointer=self._memory,
            pre_model_hook=self._pre_model_hook,
        )

        last_logged_message = None

        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=user_message)]},
            self._config,
            stream_mode="values",
        ):
            if stop_predicate():
                return ExecutionResult(response_text="[interrupted: app closed]", interrupted=True)

            last_message = step["messages"][-1]
            # Skip duplicate messages from consecutive stream events
            if last_logged_message is not None and last_message is last_logged_message:
                continue

            ctx.step_count += 1
            last_logged_message = last_message

            # Build step log entry
            step_log = {
                "step": ctx.step_count,
                "type": type(last_message).__name__,
                "phase": phase,
                "model": ctx.model,
                "content": str(last_message.content) if hasattr(last_message, "content") else None,
            }

            # Accumulate token usage
            if hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
                usage = last_message.usage_metadata
                if isinstance(usage, dict):
                    ctx.total_input_tokens += usage.get("input_tokens", 0)
                    ctx.total_output_tokens += usage.get("output_tokens", 0)
                else:
                    ctx.total_input_tokens += getattr(usage, "input_tokens", 0)
                    ctx.total_output_tokens += getattr(usage, "output_tokens", 0)

            self._service_node.get_logger().info(
                f"Step {ctx.step_count} [{phase}]: {type(last_message).__name__} - "
                f"{str(last_message.content)}"
            )

            # Capture tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                tool_calls_log = []
                for tc in last_message.tool_calls:
                    info = {"name": tc.get("name", "unknown"), "args": tc.get("args", {})}
                    tool_calls_log.append(info)
                    self._service_node.get_logger().info(
                        f"  → Tool call: {info['name']} with args: {info['args']}"
                    )
                step_log["tool_calls"] = tool_calls_log

            # Capture tool responses; track execution failures
            if isinstance(last_message, ToolMessage):
                tool_responses = []
                for msg in reversed(step["messages"]):
                    if isinstance(msg, ToolMessage):
                        tool_responses.insert(0, {
                            "tool_call_id": getattr(msg, "tool_call_id", "unknown"),
                            "name": getattr(msg, "name", "unknown"),
                            "content": str(msg.content),
                        })
                        if phase == "executing" and getattr(msg, "name", "") in (
                            "execute_single_action", "execute_sequence"
                        ):
                            try:
                                data = json.loads(msg.content)
                                phase_controller.record_exec_result(
                                    bool(data.get("success", True))
                                )
                            except json.JSONDecodeError:
                                pass
                    elif not isinstance(msg, ToolMessage):
                        break

                if len(tool_responses) > 1:
                    step_log["tool_responses"] = tool_responses
                    self._service_node.get_logger().info(
                        f"  → Received {len(tool_responses)} tool responses"
                    )
                    for i, resp in enumerate(tool_responses, 1):
                        self._service_node.get_logger().info(f"     {i}. {resp['content']}")

            ctx.step_details.append(step_log)

            # Finished when the LLM produces a plain AIMessage (no pending tool calls)
            if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
                self._service_node.get_logger().info(
                    f"Agent completed after {ctx.step_count} steps [{phase}] — "
                    f"tokens in/out: {ctx.total_input_tokens}/{ctx.total_output_tokens}"
                )
                return ExecutionResult(response_text=str(last_message.content))

        # Stream ended without a clean AIMessage (should not normally happen)
        return ExecutionResult(response_text="")
