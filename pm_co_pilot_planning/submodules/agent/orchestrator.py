from rclpy.node import Node
from langgraph.checkpoint.memory import MemorySaver

from pm_co_pilot_planning.submodules.langchain.tools.RsapTools import RsapTools
from pm_co_pilot_planning.submodules.langchain.tools.AssemblyKnowledgeTools import AssemblyKnowledgeTools
from pm_co_pilot_planning.submodules.langchain.tools.KnowledgeTools import KnowledgeTools
from pm_co_pilot_planning.submodules.langchain.LLMConfig import LLMConfig

from .phase_controller import PhaseController
from .interaction_logger import InteractionLogger
from .execution_monitor import ExecutionMonitor
from .learning_manager import LearningManager
from .memory_persistence import MemoryPersistence


class Agent:
    """
    Coordinates all agent components. Exposes the same public API as the
    original Agent class so PmCoPilotPlanningApp.py requires zero changes:
      - handle_user_input(message)
      - save_interaction_log(task_success, comment)
      - interaction_log  (read-only property)
      - stop_requested   (bool, set by app on close)
    """

    def __init__(self, service_node: Node, thread_id: str, rsap_instance=None):
        # ── Tool instances ─────────────────────────────────────────────────
        if rsap_instance:
            assembly_knowledge = AssemblyKnowledgeTools(
                service_node, rsap_instance=rsap_instance
            )
            tools_instance = RsapTools(
                service_node, rsap_instance=rsap_instance,
                assembly_knowledge=assembly_knowledge
            )
            self.rsap_instance = rsap_instance
        else:
            assembly_knowledge = AssemblyKnowledgeTools(service_node)
            tools_instance = RsapTools(service_node, assembly_knowledge=assembly_knowledge)
            self.rsap_instance = None

        self._tools_instance = tools_instance

        knowledge_tools = KnowledgeTools(service_node)

        # ── Model configs ──────────────────────────────────────────────────
        planner_config = LLMConfig("planner")
        executor_config = LLMConfig("executor")
        learner_config = LLMConfig("learner")

        self.model_name = planner_config.model
        self.model_provider = planner_config.model_provider
        self.model_configs = {
            "planning":  {"name": planner_config.model, "provider": planner_config.model_provider},
            "executing": {"name": executor_config.model, "provider": executor_config.model_provider},
            "escalated": {"name": planner_config.model, "provider": planner_config.model_provider},
            "learning":  {"name": learner_config.model, "provider": learner_config.model_provider},
        }

        # ── Tool lists ─────────────────────────────────────────────────────
        planner_tools = [
            # Domain knowledge
            knowledge_tools.query_assembly_knowledge_tool,
            # Assembly DB
            assembly_knowledge.list_available_components_tool,
            assembly_knowledge.get_component_description_tool,
            assembly_knowledge.list_available_assemblies_tool,
            assembly_knowledge.get_assembly_description_tool,
            # Live scene
            assembly_knowledge.list_objects_in_scene_tool,
            assembly_knowledge.get_object_frames_tool,
            assembly_knowledge.get_frames_in_scene_tool,
            # Sequence queries
            tools_instance.get_action_at_index_tool,
            tools_instance.get_sequence_summary_tool,
            tools_instance.get_action_parameters_tool,
            # Service discovery
            tools_instance.get_available_services_tool,
            tools_instance.get_service_parameters_tool,
            # Batch building
            tools_instance.build_sequence_from_plan_tool,
            tools_instance.load_and_modify_sequence_tool,
            # Atomic edits
            tools_instance.add_service_to_sequence_tool,
            tools_instance.add_user_interaction_tool,
            tools_instance.set_action_parameters_tool,
            tools_instance.delete_action_tool,
            tools_instance.move_action_tool,
            # Execution
            tools_instance.execute_single_action_tool,
            # Persistence
            tools_instance.save_sequence_tool,
            tools_instance.load_sequence_tool,
            tools_instance.clear_sequence_tool,
            # Heavy
            tools_instance.get_action_list_tool,
        ]

        executor_tools = [
            tools_instance.execute_single_action_tool,
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
            assembly_knowledge.get_frames_in_scene_tool,
            assembly_knowledge.list_available_components_tool,
            assembly_knowledge.get_component_description_tool,
            assembly_knowledge.list_available_assemblies_tool,
            assembly_knowledge.get_assembly_description_tool,
        ]

        learning_tools = [
            knowledge_tools.query_assembly_knowledge_tool,
            knowledge_tools.record_knowledge_tool,
            knowledge_tools.confirm_knowledge_tool,
            knowledge_tools.contradict_knowledge_tool,
            tools_instance.get_service_parameters_tool,
        ]

        planner_model = planner_config.llm.bind_tools(planner_tools, parallel_tool_calls=True)
        executor_model = executor_config.llm.bind_tools(executor_tools, parallel_tool_calls=True)
        learning_model = learner_config.llm.bind_tools(learning_tools, parallel_tool_calls=True)

        memory = MemorySaver()
        langgraph_config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}

        # ── Components ─────────────────────────────────────────────────────
        self._phase = PhaseController()
        self._logger = InteractionLogger()
        self._monitor = ExecutionMonitor(
            assembly_knowledge=assembly_knowledge,
            planner_system_prompt=planner_config.system_prompt,
            executor_system_prompt=executor_config.system_prompt,
            executor_context_window=30,
            planner_model=planner_model,
            executor_model=executor_model,
            tools=planner_tools,
            memory=memory,
            langgraph_config=langgraph_config,
            service_node=service_node,
        )
        self._learning = LearningManager(
            learning_model=learning_model,
            learning_tools=learning_tools,
            knowledge_tools=knowledge_tools,
            service_node=service_node,
            system_prompt=learner_config.system_prompt,
            rsap_instance=rsap_instance,
        )
        self._persistence = MemoryPersistence(service_node)

        self.service_node = service_node
        self.stop_requested: bool = False
        self.record_knowledge: bool = False

        service_node.get_logger().info(
            f"AgentOrchestrator initialized — "
            f"planner: {planner_config.model}, executor: {executor_config.model}, "
            f"learner: {learner_config.model}"
        )

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def interaction_log(self) -> list:
        return self._logger.interactions

    @property
    def current_phase(self) -> str:
        return self._phase.current_phase

    def handle_user_input(self, user_message: str) -> str:
        """
        Main entry point. Replaces the original recursive handle_user_input with
        a deterministic loop:
          1. Detect phase from user message
          2. Run one LLM interaction
          3. Decide: escalate / continue / done
          4. After loop: run learning nudge if any execution occurred
        """
        self._phase.set_from_message(user_message)
        if self._phase.current_phase == "executing":
            self._tools_instance.last_executed_user_index = 0
        session_had_execution = False
        pending = user_message
        final_response = ""

        while pending is not None:
            if self.stop_requested:
                # Log the interrupted interaction and flush to disk immediately
                ctx = self._logger.start(
                    pending,
                    self._phase.current_phase,
                    self.model_configs.get(self._phase.current_phase, {}).get("name", "unknown"),
                )
                self._logger.finish(ctx, "[interrupted: app closed]")
                self._run_learning_if_needed(session_had_execution)
                self.save_interaction_log()
                return "Agent stopped: application was closed."

            phase = self._phase.current_phase
            model_name = self.model_configs.get(phase, {}).get("name", "unknown")

            self.service_node.get_logger().info(
                f"Starting agent execution [{phase} → {model_name}] for: {pending[:80]}..."
            )

            ctx = self._logger.start(pending, phase, model_name)
            result = self._monitor.run_once(
                user_message=pending,
                phase=phase,
                ctx=ctx,
                phase_controller=self._phase,
                stop_predicate=lambda: self.stop_requested,
            )

            # Handle mid-stream stop
            if result.interrupted:
                self._logger.finish(ctx, result.response_text)
                self._run_learning_if_needed(session_had_execution)
                self.save_interaction_log()
                return "Agent stopped: application was closed."

            self._logger.finish(ctx, result.response_text)

            if phase in ("executing", "escalated"):
                session_had_execution = True

            # Let PhaseController decide what to do next
            total_actions = len(self.rsap_instance.action_list) if self.rsap_instance else 0
            self.service_node.get_logger().info(
                f"decide_next inputs: total_actions={total_actions}, "
                f"last_executed_index={self._tools_instance.last_executed_user_index}, "
                f"full_execution_requested={self._phase.full_execution_requested}, "
                f"phase={self._phase.current_phase}"
            )
            decision = self._phase.decide_next(
                response_text=result.response_text,
                total_actions=total_actions,
                last_executed_index=self._tools_instance.last_executed_user_index,
            )

            if decision.kind == "escalate":
                self.service_node.get_logger().info(
                    "Executor escalating to planner for complex error diagnosis"
                )
                self._phase.mark_escalated()
                pending = decision.prompt
            elif decision.kind == "continue":
                self.service_node.get_logger().info(
                    f"Auto-continuing execution: {decision.prompt[:60]}..."
                )
                pending = decision.prompt  # phase stays "executing"
            else:
                final_response = result.response_text
                pending = None

        self._run_learning_if_needed(session_had_execution)
        self._phase.reset_to_planning()
        return final_response

    def _run_learning_if_needed(self, session_had_execution: bool) -> None:
        if session_had_execution and self.record_knowledge:
            learner_model_name = self.model_configs.get("learning", {}).get("name", "unknown")
            self._learning.run(
                interaction_logger=self._logger,
                executed_services=self._get_executed_services(),
                learner_model_name=learner_model_name,
            )

    def save_interaction_log(self, task_success=None, comment=None):
        self._persistence.save_session_log(
            interaction_log=self._logger.interactions,
            model_name=self.model_name,
            model_configs=self.model_configs,
            rsap_instance=self.rsap_instance,
            task_success=task_success,
            comment=comment,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_executed_services(self) -> list:
        """Return deduplicated service client names from the current action sequence."""
        if not self.rsap_instance:
            return []
        seen = set()
        result = []
        for action in self.rsap_instance.action_list:
            client = getattr(action, "client", None)
            if client and client not in seen:
                seen.add(client)
                result.append(client)
        return result
