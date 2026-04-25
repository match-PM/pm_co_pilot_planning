from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage


class ConsolidationManager:
    """
    Periodic KB clean-up via a single LLM call (no tool use).

    The consolidator receives the full knowledge base as YAML and returns a
    cleaned version: false contradictions resolved, redundant entries merged,
    complementary use cases separated, and confidences adjusted according to
    confirmation/contradiction counts.

    Triggered from the orchestrator every N learning sessions.
    """

    def __init__(self, consolidator_model, knowledge_tools, service_node, system_prompt: str):
        self._model = consolidator_model
        self._knowledge_tools = knowledge_tools
        self._service_node = service_node
        self._system_prompt = system_prompt

    def run(self, interaction_logger, current_counter: int, model_name: str) -> None:
        """Single LLM call that produces a consolidated KB and writes it back."""
        self._service_node.get_logger().info(
            f"KB consolidation triggered (update_counter={current_counter})"
        )

        try:
            kb_yaml = self._knowledge_tools.load_full_knowledge_yaml()
        except Exception as e:
            self._service_node.get_logger().warning(
                f"Consolidation skipped: could not load KB — {e}"
            )
            return

        start = datetime.now()
        input_tokens = 0
        output_tokens = 0
        result_text = ""

        try:
            response = self._model.invoke([
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=kb_yaml),
            ])
            result_text = str(response.content)

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
                else:
                    input_tokens = getattr(usage, "input_tokens", 0)
                    output_tokens = getattr(usage, "output_tokens", 0)

        except Exception as e:
            self._service_node.get_logger().warning(
                f"Consolidation LLM call failed (non-critical): {e}"
            )
            self._log(interaction_logger, start, model_name, f"[LLM error: {e}]",
                      input_tokens, output_tokens)
            return

        end = datetime.now()
        self._service_node.get_logger().info(
            f"Consolidation response received ({len(result_text)} chars, "
            f"tokens in/out: {input_tokens}/{output_tokens})"
        )

        if result_text.strip() == "No consolidation needed.":
            self._service_node.get_logger().info(
                "Consolidation: no changes needed — advancing counter."
            )
            self._advance_counter(current_counter)
            self._log(interaction_logger, start, model_name, result_text, input_tokens, output_tokens, end)
            return

        # Strip optional ```yaml fences
        cleaned = result_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Early YAML validation — surface the exact error line before handing off
        import yaml as _yaml
        try:
            _yaml.safe_load(cleaned)
        except _yaml.YAMLError as e:
            self._service_node.get_logger().warning(
                f"Consolidation: LLM produced invalid YAML — counter NOT advanced.\n"
                f"Parser error: {e}\n"
                f"Hint: note fields with colons must use block scalars (note: |)."
            )
            self._log(interaction_logger, start, model_name,
                      f"[yaml parse error: {e}]\n\n{result_text}", input_tokens, output_tokens, end)
            return

        try:
            self._knowledge_tools.replace_knowledge(cleaned, last_consolidated_at_count=current_counter)
            self._service_node.get_logger().info("Consolidation: KB successfully replaced.")
        except (ValueError, Exception) as e:
            self._service_node.get_logger().warning(
                f"Consolidation: KB write failed — counter NOT advanced. Error: {e}"
            )
            self._log(interaction_logger, start, model_name,
                      f"[write error: {e}]\n\n{result_text}", input_tokens, output_tokens, end)
            return

        self._log(interaction_logger, start, model_name, result_text, input_tokens, output_tokens, end)

    def _advance_counter(self, current_counter: int) -> None:
        """Update last_consolidated_at_count without replacing KB content."""
        try:
            kb_yaml = self._knowledge_tools.load_full_knowledge_yaml()
            # Re-use replace_knowledge with the unchanged content just to bump the counter
            self._knowledge_tools.replace_knowledge(kb_yaml, last_consolidated_at_count=current_counter)
        except Exception as e:
            self._service_node.get_logger().warning(
                f"Consolidation: could not advance counter: {e}"
            )

    def _log(self, interaction_logger, start, model_name, response_text,
             input_tokens, output_tokens, end=None) -> None:
        end = end or datetime.now()
        interaction_logger.append_raw({
            "timestamp": start.isoformat(),
            "timestamp_end": end.isoformat(),
            "execution_time_seconds": (end - start).total_seconds(),
            "user_message": "[post-learning KB consolidation]",
            "agent_response": response_text,
            "phase": "consolidation",
            "model": model_name,
            "steps": 1,
            "step_details": [],
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
        })
