import json
import os
from datetime import datetime


class MemoryPersistence:
    """Handles saving session logs to disk. No in-memory state."""

    def __init__(self, service_node):
        self._service_node = service_node

    def save_session_log(
        self,
        interaction_log: list,
        model_name: str,
        model_configs: dict,
        rsap_instance,
        task_success=None,
        comment=None,
    ):
        if not interaction_log:
            self._service_node.get_logger().info("No interactions to save")
            return

        sequence_name = None
        if rsap_instance and hasattr(rsap_instance, "rsap_file_manager"):
            folder_path = rsap_instance.rsap_file_manager.get_folder_path()
            try:
                sequence_name = rsap_instance.rsap_file_manager.get_sequence_name()
            except Exception as e:
                self._service_node.get_logger().warning(f"Could not get sequence name: {e}")
        else:
            folder_path = "/home/match-pm/Desktop"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
        filename = f"copilot_log_{timestamp}_{safe_model_name}.json"
        filepath = os.path.join(folder_path, filename)

        total_interactions = len(interaction_log)
        total_steps = sum(log["steps"] for log in interaction_log)
        total_tokens = sum(log["tokens"]["total"] for log in interaction_log)
        total_input_tokens = sum(log["tokens"]["input"] for log in interaction_log)
        total_output_tokens = sum(log["tokens"]["output"] for log in interaction_log)

        final_sequence = []
        if rsap_instance:
            try:
                for idx, action in enumerate(rsap_instance.action_list):
                    action_info = {
                        "index": idx + 1,
                        "name": action.get_name() if hasattr(action, "get_name") else str(action),
                        "type": type(action).__name__,
                        "is_active": action.is_active() if hasattr(action, "is_active") else True,
                    }
                    if hasattr(action, "client"):
                        action_info["client"] = action.client
                    if hasattr(action, "get_request_as_ordered_dict"):
                        action_info["parameters"] = dict(action.get_request_as_ordered_dict())
                    final_sequence.append(action_info)
            except Exception as e:
                self._service_node.get_logger().warning(f"Could not capture final sequence: {e}")

        log_data = {
            "model": model_name,
            "models": model_configs,
            "task_success": task_success,
            "comment": comment,
            "sequence_name": sequence_name,
            "session_start": interaction_log[0]["timestamp"] if interaction_log else None,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_interactions": total_interactions,
                "total_steps": total_steps,
                "total_tokens": total_tokens,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
            },
            "interactions": interaction_log,
            "final_sequence": {
                "total_actions": len(final_sequence),
                "actions": final_sequence,
            },
        }

        try:
            with open(filepath, "w") as f:
                json.dump(log_data, f, indent=2)
            self._service_node.get_logger().info(f"Interaction log saved to: {filepath}")
            self._service_node.get_logger().info(
                f"Session summary: {total_interactions} interactions, "
                f"{total_steps} steps, {total_tokens} tokens"
            )
        except Exception as e:
            self._service_node.get_logger().error(f"Failed to save interaction log: {e}")
