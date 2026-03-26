import getpass
import os
import json
import pickle
from datetime import datetime

from rclpy.node import Node

from langchain_core.tools import BaseTool
from langchain_core.tools import tool

from langchain_core.messages import HumanMessage, AIMessage, FunctionMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent


from langchain.chat_models import init_chat_model
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from collections import defaultdict

from pm_co_pilot_planning.submodules.langchain.tools.RsapTools import RsapTools
from pm_co_pilot_planning.submodules.langchain.tools.AssemblyKnowledgeTools import AssemblyKnowledgeTools
from pm_co_pilot_planning.submodules.langchain.tools.KnowledgeTools import KnowledgeTools
from pm_co_pilot_planning.submodules.langchain.LLMConfig import LLMConfig


# Keywords that indicate the user wants to execute (not plan/build)
_EXECUTE_KEYWORDS = ["execute", "run the sequence", "run it", "start execution", "run all"]


class Agent:
    """
    The Agent class is responsible for managing the interaction with the LLM.
    It uses a dual-model architecture:
      - Planner (expensive model): for building sequences, modifying, and complex error diagnosis
      - Executor (cheap model): for running actions and handling routine errors
    A pre_model_hook windows the context during execution to avoid O(n^2) token growth.
    """

    def __init__(self, service_node: Node, thread_id: str, rsap_instance=None):

        # Create AssemblyKnowledgeTools first so it can be passed to RsapTools for state-diff
        if rsap_instance:
            assembly_knowledge = AssemblyKnowledgeTools(service_node, rsap_instance=rsap_instance)
            tools_instance = RsapTools(service_node, rsap_instance=rsap_instance, assembly_knowledge=assembly_knowledge)
            self.rsap_instance = rsap_instance
        else:
            assembly_knowledge = AssemblyKnowledgeTools(service_node)
            tools_instance = RsapTools(service_node, assembly_knowledge=assembly_knowledge)
            self.rsap_instance = None

        # Store reference for scene injection in pre_model_hook
        self._assembly_knowledge = assembly_knowledge

        knowledge_tools = KnowledgeTools(service_node)

        # Initialize interaction log
        self.interaction_log = []

        # ── Load dual model configs ─────────────────────────────────────────────
        planner_config = LLMConfig('planner')
        executor_config = LLMConfig('executor')

        # Store model info for logging
        self.model_name = planner_config.model  # primary model for log filenames
        self.model_provider = planner_config.model_provider
        self.model_configs = {
            "planning": {"name": planner_config.model, "provider": planner_config.model_provider},
            "executing": {"name": executor_config.model, "provider": executor_config.model_provider},
            "escalated": {"name": planner_config.model, "provider": planner_config.model_provider},
        }

        # ── Phase tracking ──────────────────────────────────────────────────────
        self.current_phase = "planning"  # "planning" | "executing" | "escalated"
        self.consecutive_exec_failures = 0  # track execution failures for auto-escalate
        self.stop_requested = False  # set to True to abort agent loop on app close


        # ── System prompts (injected via pre_model_hook) ────────────────────────
        self._planner_system_prompt = planner_config.system_prompt
        self._executor_system_prompt = executor_config.system_prompt

        # Context window size for executor (number of recent messages to keep)
        # ~20 messages ≈ last 10 tool call/response pairs, enough for 2-3 retry cycles
        self.executor_context_window = 20

        # ── Full tool set for planner ───────────────────────────────────────────
        self.tools = [
            # ── Domain knowledge (use first to retrieve learned rules) ───────────
            knowledge_tools.query_assembly_knowledge_tool,
            # knowledge_tools.get_similar_assembly_example_tool,
            knowledge_tools.record_knowledge_tool,

            # ── Assembly knowledge (use when planning a new sequence) ────────────
            assembly_knowledge.list_available_components_tool,
            assembly_knowledge.get_component_description_tool,
            assembly_knowledge.list_available_assemblies_tool,
            assembly_knowledge.get_assembly_description_tool,

            # ── Live scene knowledge ──────────────────────────────────────────────
            assembly_knowledge.list_objects_in_scene_tool,
            assembly_knowledge.get_object_properties_tool,
            assembly_knowledge.get_object_frames_tool,
            assembly_knowledge.get_frame_properties_tool,
            assembly_knowledge.get_frames_in_scene_tool,

            # ── Efficient query tools ─────────────────────────────────────────────
            tools_instance.get_action_at_index_tool,        # For "what's at index X?"
            tools_instance.get_sequence_summary_tool,       # For "show me the sequence"
            tools_instance.get_action_parameters_tool,      # For "what are the current parameters?"

            # ── Service/Action discovery ──────────────────────────────────────────
            tools_instance.get_available_services_tool,
            tools_instance.get_service_parameters_tool,
            # tools_instance.get_parameter_value_recommendations_tool,

            # ── Batch sequence building (preferred for new complete sequences) ────
            tools_instance.build_sequence_from_plan_tool,
            tools_instance.load_and_modify_sequence_tool,

            # ── Atomic sequence building (for additions / edits) ──────────────────
            tools_instance.add_service_to_sequence_tool,
            tools_instance.add_user_interaction_tool,

            # ── Modifying sequence ────────────────────────────────────────────────
            tools_instance.set_action_parameters_tool,
            tools_instance.delete_action_tool,
            tools_instance.move_action_tool,

            # ── Execution ─────────────────────────────────────────────────────────
            tools_instance.execute_sequence_tool,
            tools_instance.execute_single_action_tool,

            # ── Sequence persistence ──────────────────────────────────────────────
            tools_instance.save_sequence_tool,
            tools_instance.load_sequence_tool,
            tools_instance.clear_sequence_tool,

            # ── Heavy (use sparingly) ─────────────────────────────────────────────
            tools_instance.get_action_list_tool,
        ]

        # ── Minimal tool subset for executor ────────────────────────────────────
        # Note: list_objects_in_scene and get_object_properties are removed because
        # the scene summary is injected automatically via pre_model_hook.
        self.executor_tools = [
            tools_instance.execute_single_action_tool,
            tools_instance.execute_sequence_tool,
            tools_instance.get_action_at_index_tool,
            tools_instance.get_action_parameters_tool,
            tools_instance.set_action_parameters_tool,
            tools_instance.get_sequence_summary_tool,
            tools_instance.get_available_services_tool,
            tools_instance.get_service_parameters_tool,
            tools_instance.add_service_to_sequence_tool,
            knowledge_tools.query_assembly_knowledge_tool,
            knowledge_tools.record_knowledge_tool,
            assembly_knowledge.get_object_frames_tool,
            assembly_knowledge.get_frame_properties_tool,
            assembly_knowledge.get_frames_in_scene_tool,
            assembly_knowledge.list_available_components_tool,
            assembly_knowledge.get_component_description_tool,
            assembly_knowledge.list_available_assemblies_tool,
            assembly_knowledge.get_assembly_description_tool,
        ]

        # ── Bind tools to each model ───────────────────────────────────────────
        # Planner gets all tools, executor gets the minimal subset
        self.planner_model = planner_config.llm.bind_tools(self.tools, parallel_tool_calls=True)
        self.executor_model = executor_config.llm.bind_tools(self.executor_tools, parallel_tool_calls=True)

        self.memory = MemorySaver()
        self.config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100
        }
        self.service_node = service_node

        # When  app starts, try to load any previously saved memory state
        # self.load_memory()

        self.service_node.get_logger().info(
            f"Agent initialized — planner: {planner_config.model}, executor: {executor_config.model}"
        )

    # ── Dynamic model selection ─────────────────────────────────────────────────

    def _select_model(self, state, runtime=None):
        """Return the executor model during execution, planner model otherwise."""
        if self.current_phase == "executing":
            return self.executor_model
        return self.planner_model

    # ── Pre-model hook for context windowing ────────────────────────────────────

    def _pre_model_hook(self, state):
        """Control what messages the LLM sees.

        - Planning/escalated: full conversation history (planner needs full context)
        - Executing: windowed to last N messages (avoids O(n^2) token growth)

        System prompts are injected here since we use prompt=None in create_executor.
        Scene state is injected as a SystemMessage after the system prompt.
        """
        messages = state["messages"]
        scene_msg = SystemMessage(content=self._assembly_knowledge.get_compact_scene_summary())

        if self.current_phase != "executing":
            # Planning: LLM sees everything
            return {
                "llm_input_messages": [
                    SystemMessage(content=self._planner_system_prompt),
                    scene_msg,
                ] + list(messages)
            }

        # Execution: window to recent messages only
        # Always include the first HumanMessage (the execution instruction)
        first_human = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                first_human = msg
                break

        recent = list(messages[-self.executor_context_window:])

        # ToolMessages must follow an AIMessage with tool_calls.
        # If the window cut leaves orphaned ToolMessages at the start, drop them.
        while recent and isinstance(recent[0], ToolMessage):
            recent.pop(0)

        windowed = [SystemMessage(content=self._executor_system_prompt), scene_msg]
        if first_human and first_human not in recent:
            windowed.append(first_human)
        windowed.extend(recent)

        return {"llm_input_messages": windowed}

    # ── Executor creation ───────────────────────────────────────────────────────

    def create_executor(self):
        return create_react_agent(
            model=self._select_model,
            tools=self.tools,
            checkpointer=self.memory,
            pre_model_hook=self._pre_model_hook,
            # prompt=None: system prompts are handled in _pre_model_hook
        )

    def _detect_phase(self, user_message: str) -> str:
        """Detect whether this message should use the executor or planner model."""
        msg_lower = user_message.lower()
        if any(kw in msg_lower for kw in _EXECUTE_KEYWORDS):
            return "executing"
        return "planning"

    def handle_user_input(self, user_message: str) -> str:
        """
        Non-streaming approach (returns final string) with debug logging and token tracking.
        Automatically selects planner vs executor model based on user intent.
        """
        # Detect phase from user message (unless already escalated)
        if self.current_phase != "escalated":
            new_phase = self._detect_phase(user_message)
            if new_phase != "executing" and self.current_phase == "executing":
                # if switching out of execution, reset failure counter
                self.consecutive_exec_failures = 0
            self.current_phase = new_phase

        phase_model = self.model_configs.get(self.current_phase, {}).get("name", "unknown")
        self.service_node.get_logger().info(
            f"Starting agent execution [{self.current_phase} → {phase_model}] for: {user_message[:100]}..."
        )

        agent_executor = self.create_executor()

        # Track token usage, step details, and timing
        interaction_start = datetime.now()
        total_input_tokens = 0
        total_output_tokens = 0
        step_details = []  # Store detailed step information
        last_logged_message = None  # Track last message to avoid duplicates

        try:
            # Stream to see each step
            step_count = 0
            for step in agent_executor.stream(
                {"messages": [HumanMessage(content=user_message)]},
                self.config,
                stream_mode="values"
            ):
                if self.stop_requested:
                    self.service_node.get_logger().info("Agent stopped: app was closed.")
                    interaction_end = datetime.now()
                    self.interaction_log.append({
                        "timestamp": interaction_start.isoformat(),
                        "timestamp_end": interaction_end.isoformat(),
                        "execution_time_seconds": (interaction_end - interaction_start).total_seconds(),
                        "user_message": user_message,
                        "agent_response": "[interrupted: app closed]",
                        "phase": self.current_phase,
                        "model": phase_model,
                        "steps": step_count,
                        "step_details": step_details,
                        "tokens": {
                            "input": total_input_tokens,
                            "output": total_output_tokens,
                            "total": total_input_tokens + total_output_tokens
                        }
                    })
                    self.current_phase = "planning"
                    # Save immediately — cleanup() may have already run by the time we get here
                    self.save_interaction_log()
                    return "Agent stopped: application was closed."

                last_message = step["messages"][-1]

                # Skip duplicate messages (same message appearing in consecutive stream events)
                if last_logged_message is not None and last_message is last_logged_message:
                    continue
                
                step_count += 1
                last_logged_message = last_message

                # Create step log entry with model/phase tracking
                step_log = {
                    "step": step_count,
                    "type": type(last_message).__name__,
                    "phase": self.current_phase,
                    "model": self.model_configs.get(self.current_phase, {}).get("name", "unknown"),
                    "content": str(last_message.content) if hasattr(last_message, 'content') else None
                }

                # Extract token usage if available
                if hasattr(last_message, 'usage_metadata') and last_message.usage_metadata:
                    usage = last_message.usage_metadata
                    if isinstance(usage, dict):
                        total_input_tokens += usage.get('input_tokens', 0)
                        total_output_tokens += usage.get('output_tokens', 0)
                    else:
                        total_input_tokens += getattr(usage, 'input_tokens', 0)
                        total_output_tokens += getattr(usage, 'output_tokens', 0)

                # Log each step for debugging
                if hasattr(last_message, 'content'):
                    self.service_node.get_logger().info(
                        f"Step {step_count} [{self.current_phase}]: {type(last_message).__name__} - {str(last_message.content)}"
                    )

                # Capture tool calls (when agent calls tools)
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    tool_calls_log = []
                    for tool_call in last_message.tool_calls:
                        tool_call_info = {
                            "name": tool_call.get('name', 'unknown'),
                            "args": tool_call.get('args', {})
                        }
                        tool_calls_log.append(tool_call_info)
                        self.service_node.get_logger().info(
                            f"  → Tool call: {tool_call_info['name']} with args: {str(tool_call_info['args'])}"
                        )
                    step_log["tool_calls"] = tool_calls_log

                # Capture tool responses (may be multiple in parallel calls)
                if isinstance(last_message, ToolMessage):
                    tool_responses = []

                    # Go backwards to find all ToolMessages added in this step
                    for msg in reversed(step["messages"]):
                        if isinstance(msg, ToolMessage):
                            tool_responses.insert(0, {
                                "tool_call_id": getattr(msg, 'tool_call_id', 'unknown'),
                                "name": getattr(msg, 'name', 'unknown'),
                                "content": str(msg.content)
                            })
                            
                            # Auto-escalation trigger based on continuous execution parsing
                            if self.current_phase == "executing" and getattr(msg, 'name', 'unknown') in ["execute_single_action", "execute_sequence"]:
                                try:
                                    response_data = json.loads(msg.content)
                                    if response_data.get("success") is False:
                                        self.consecutive_exec_failures += 1
                                    else:
                                        self.consecutive_exec_failures = 0
                                except json.JSONDecodeError:
                                    pass

                        elif not isinstance(msg, ToolMessage):
                            break

                    if len(tool_responses) > 1:
                        step_log["tool_responses"] = tool_responses
                        self.service_node.get_logger().info(
                            f"  → Received {len(tool_responses)} tool responses"
                        )
                        for i, resp in enumerate(tool_responses, 1):
                            self.service_node.get_logger().info(
                                f"     {i}. {resp['content']}"
                            )

                step_details.append(step_log)

                # Check if we're done (AIMessage with no tool calls)
                if isinstance(last_message, AIMessage) and not getattr(last_message, 'tool_calls', None):
                    interaction_end = datetime.now()
                    execution_time = (interaction_end - interaction_start).total_seconds()

                    self.service_node.get_logger().info(
                        f"Agent completed after {step_count} steps [{self.current_phase} → {phase_model}]"
                    )
                    self.service_node.get_logger().info(
                        f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
                    )
                    self.service_node.get_logger().info(f"Execution time: {execution_time:.2f} seconds")

                    # Log this interaction with detailed steps and timing
                    self.interaction_log.append({
                        "timestamp": interaction_start.isoformat(),
                        "timestamp_end": interaction_end.isoformat(),
                        "execution_time_seconds": execution_time,
                        "user_message": user_message,
                        "agent_response": last_message.content,
                        "phase": self.current_phase,
                        "model": phase_model,
                        "steps": step_count,
                        "step_details": step_details,
                        "tokens": {
                            "input": total_input_tokens,
                            "output": total_output_tokens,
                            "total": total_input_tokens + total_output_tokens
                        }
                    })

                    response_text = last_message.content

                    # ── Escalation: executor couldn't fix an error ──────────
                    if self.current_phase == "executing" and ("ESCALATE:" in response_text or self.consecutive_exec_failures >= 3):
                        self.consecutive_exec_failures = 0  # reset for next time
                        self.service_node.get_logger().info(
                            "Executor escalating to planner for complex error diagnosis (implicit or explicit)"
                        )
                        self.current_phase = "escalated"
                        return self.handle_user_input(
                            f"The execution monitor could not resolve this error and escalated to you.\n"
                            f"Recent response: {response_text}\n"
                            f"Please diagnose the root cause using query_assembly_knowledge and fix the sequence. "
                            f"After fixing, continue execution from the failed action."
                        )

                    # ── Post-execution learning nudge ───────────────────────
                    completed_phase = self.current_phase
                    self.current_phase = "planning"

                    if completed_phase == "executing":
                        self.service_node.get_logger().info(
                            "Post-execution learning nudge: asking agent to record discovered knowledge."
                        )
                        try:
                            self.handle_user_input(
                                "Execution completed successfully. "
                                "For each service used that has empty preconditions, postconditions, or parameters "
                                "in the knowledge base, fill them in now using record_knowledge: "
                                "for parameters call get_service_parameters first, then record each parameter; "
                                "for preconditions/postconditions infer the fact tokens from observed behavior. "
                                "Also record any other new discoveries (constraints, lessons learned). "
                                "If nothing is missing or new, respond with 'No new knowledge to record.'"
                            )
                        except Exception as learn_err:
                            self.service_node.get_logger().warning(
                                f"Post-execution learning call failed (non-critical): {learn_err}"
                            )

                    return response_text

            # Fallback: get the last message
            messages = agent_executor.invoke({"messages": [HumanMessage(content=user_message)]}, self.config)
            ai_message = messages["messages"][-1]
            response = ai_message.content
            self.service_node.get_logger().info(f"Agent messages: {messages}")
            self.service_node.get_logger().info(
                f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
            )
            self.current_phase = "planning"
            return response

        except Exception as e:
            self.service_node.get_logger().error(f"Agent execution error after {step_count} steps: {e}")
            if total_input_tokens > 0 or total_output_tokens > 0:
                self.service_node.get_logger().info(
                    f"Token usage before error - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_input_tokens + total_output_tokens}"
                )
            self.current_phase = "planning"
            raise

    
    def save_memory(self):
        """
        Save the current memory state to a file.
        """
        data_to_save = self.convert_defaultdict_to_dict(self.memory.storage)
        with open("chat_history.pkl", "wb") as f:
            self.service_node.get_logger().info(f"Saving memory state: {data_to_save}")
            pickle.dump(data_to_save, f)
            self.service_node.get_logger().info("Memory state saved.")

    def load_memory(self):
        """
        Load the memory state from a file.
        """
        try:
            with open("chat_history.pkl", "rb") as f:
                loaded_data = pickle.load(f)
                # now loaded_data is just a nested dict
                self.memory.storage = loaded_data
                self.service_node.get_logger().info("Memory state loaded.")
        except FileNotFoundError:
            self.service_node.get_logger().info("No saved memory state found.")
            pass

    

    # a helper function to recursively convert defaultdict -> dict
    def convert_defaultdict_to_dict(self, obj):
        if isinstance(obj, defaultdict):
            # convert its contents recursively into a normal dict
            obj = {k: self.convert_defaultdict_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            # just handle nested dict
            obj = {k: self.convert_defaultdict_to_dict(v) for k, v in obj.items()}
        return obj

    def save_interaction_log(self, task_success=None, comment=None):
        """Save the interaction log to a JSON file with timestamp.
        
        Args:
            task_success: Optional bool indicating if the user confirmed task was successful (True/False/None)
            comment: Optional string with user's comment about the task execution
        """
        if not self.interaction_log:
            self.service_node.get_logger().info("No interactions to save")
            return
        
        # Get folder path from RSAP instance
        if self.rsap_instance and hasattr(self.rsap_instance, 'rsap_file_manager'):
            folder_path = self.rsap_instance.rsap_file_manager.get_folder_path()
        else:
            folder_path = "/home/match-pm/Desktop"  # Default fallback
        
        # Create filename with timestamp and model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename (replace / and : with _)
        safe_model_name = self.model_name.replace('/', '_').replace(':', '_').replace('-', '_')
        filename = f"copilot_log_{timestamp}_{safe_model_name}.json"
        filepath = os.path.join(folder_path, filename)
        
        # Calculate totals
        total_interactions = len(self.interaction_log)
        total_steps = sum(log["steps"] for log in self.interaction_log)
        total_tokens = sum(log["tokens"]["total"] for log in self.interaction_log)
        total_input_tokens = sum(log["tokens"]["input"] for log in self.interaction_log)
        total_output_tokens = sum(log["tokens"]["output"] for log in self.interaction_log)
        
        # Capture current sequence state
        final_sequence = []
        if self.rsap_instance:
            try:
                for idx, action in enumerate(self.rsap_instance.action_list):
                    action_info = {
                        "index": idx + 1,  # 1-based for consistency with GUI
                        "name": action.get_name() if hasattr(action, 'get_name') else str(action),
                        "type": type(action).__name__,
                        "is_active": action.is_active() if hasattr(action, 'is_active') else True
                    }
                    # Add client info if available
                    if hasattr(action, 'client'):
                        action_info["client"] = action.client
                    # Add parameters if available
                    if hasattr(action, 'get_request_as_ordered_dict'):
                        action_info["parameters"] = dict(action.get_request_as_ordered_dict())
                    final_sequence.append(action_info)
            except Exception as e:
                self.service_node.get_logger().warning(f"Could not capture final sequence: {e}")
        
        # Create log data structure
        log_data = {
            "model": self.model_name,
            "models": self.model_configs,
            "task_success": task_success,
            "comment": comment,
            "session_start": self.interaction_log[0]["timestamp"] if self.interaction_log else None,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_interactions": total_interactions,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            },
            "interactions": self.interaction_log,
            "final_sequence": {
                "total_actions": len(final_sequence),
                "actions": final_sequence
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            self.service_node.get_logger().info(f"Interaction log saved to: {filepath}")
            self.service_node.get_logger().info(
                f"Session summary: {total_interactions} interactions, {total_steps} steps, {total_tokens} tokens"
            )
        except Exception as e:
            self.service_node.get_logger().error(f"Failed to save interaction log: {e}")





