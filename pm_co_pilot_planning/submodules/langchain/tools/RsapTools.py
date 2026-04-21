from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from rclpy.node import Node

import yaml
import json
import copy
import threading
from typing import Optional, Dict, Any, List

from ament_index_python.packages import get_package_share_directory
from ros_sequential_action_programmer.submodules.RosSequentialActionProgrammer import RosSequentialActionProgrammer
from rosidl_runtime_py.set_message import set_message_fields

from pm_co_pilot_planning.submodules.langchain.tools._helpers import (
    ToolResponse, parse_tool_index, parse_tool_indices, to_internal_index,
    ActionHelper, ValueSetHelper,
)


# Pydantic schemas for StructuredTool inputs
class SetActionParametersInput(BaseModel):
    """Input schema for set_action_parameters tool."""
    index: int = Field(description="1-based index of the action in the sequence (GUI index)")
    parameters: Dict[str, Any] = Field(description="Dictionary of parameter key-value pairs to set. REQUIRED - must contain at least one key-value pair. Use get_action_parameters first to discover available parameter names. For nested messages like Vector3 or Quaternion, use nested dictionaries.")


class MoveActionInput(BaseModel):
    """Input schema for move_action tool."""
    old_index: int = Field(description="Current 1-based index of the action to move (GUI index)")
    new_index: int = Field(description="Target 1-based index where the action should be moved (GUI index)")


class GetSequenceSummaryInput(BaseModel):
    """Input schema for get_sequence_summary tool (no parameters required)."""
    pass


class ExecuteSequenceInput(BaseModel):
    """Input schema for execute_sequence tool."""
    start_index: Optional[int] = Field(default=0, description="0-based index to start execution from (default 0 = beginning)")


class AddServiceToSequenceInput(BaseModel):
    """Input schema for add_service_to_sequence tool."""
    service_client: str = Field(description="The ROS2 service client name (e.g., '/move_robot')")
    index: int = Field(description="1-based index where to insert the service in the sequence")
    service_type: Optional[str] = Field(default=None, description="Optional service type")
    service_name: Optional[str] = Field(default=None, description="Optional custom name for the service action")


class AddRosActionToSequenceInput(BaseModel):
    """Input schema for add_ros_action_to_sequence tool."""
    action_client: str = Field(description="The ROS2 action client name (e.g., '/navigate_to_pose')")
    index: int = Field(description="1-based index where to insert the action in the sequence")
    action_type: Optional[str] = Field(default=None, description="Optional action type")
    action_name: Optional[str] = Field(default=None, description="Optional custom name for the action")


class AddUserInteractionInput(BaseModel):
    """Input schema for add_user_interaction tool."""
    index: int = Field(description="1-based index where to insert the user interaction in the sequence")
    action_name: str = Field(description="Name for the user interaction action")
    action_description: str = Field(description="Description of what the user should do")
    interaction_mode: str = Field(default="terminal", description="Interaction mode: 'terminal' or 'gui'")


class GetParameterValueRecommendationsInput(BaseModel):
    """Input schema for get_parameter_value_recommendations tool."""
    parameter_type: Optional[str] = Field(default=None, description="Optional parameter type filter (e.g., 'string', 'str', 'uint32'). If omitted, returns all value sets.")


class ActionSpec(BaseModel):
    """Single action specification used inside BuildSequenceFromPlanInput."""
    service_client: str = Field(description="ROS2 service client name (e.g. '/pm_skills/vision_correct_frame')")
    name: str = Field(description="Display name shown in the RSAP UI (e.g. 'Correct UFC Vision 1')")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter key-value pairs to set on the action. Nested dicts for nested ROS messages. Arrays of nested messages must be lists of dicts, e.g. dispense_points: [{frame_name: 'P1', time_ms: 500.0, dispense_z_offset_mm: 0.0}] — never lists of plain strings."
    )
    service_type: Optional[str] = Field(default=None, description="Optional ROS2 service type string")


class BuildSequenceFromPlanInput(BaseModel):
    """Input schema for build_sequence_from_plan tool."""
    actions: List[ActionSpec] = Field(
        description="Ordered list of actions to add to the sequence. "
                    "They are appended in order starting at start_index."
    )
    clear_existing: bool = Field(
        default=False,
        description="If True, clear the existing sequence before building. Default False."
    )
    start_index: Optional[int] = Field(
        default=None,
        description="1-based index where insertion starts. If None, actions are appended to the end."
    )


class LoadAndModifySequenceInput(BaseModel):
    """Input schema for load_and_modify_sequence tool."""
    file_path: str = Field(
        description="Absolute path to a .rsap.json sequence file to load into the current RSAP instance."
    )


class RsapTools:
    """
    The Tools class provides a set of tools that can be used by the agent.
    Each tool is a function that performs a specific action to control the RosSequentialActionProgrammer.
    """
    def __init__(self, service_node: Node, rsap_instance=None, assembly_knowledge=None):
        self.service_node = service_node
        # Use provided RSAP instance or create a new one
        if rsap_instance:
            self.rsap = rsap_instance
        else:
            self.rsap = RosSequentialActionProgrammer(service_node)

        # Optional reference to AssemblyKnowledgeTools for state-diff on execution
        self._assembly_knowledge = assembly_knowledge

        # Add lock for sequence modification operations
        self._sequence_lock = threading.Lock()

        # Tracks the highest 1-based user index successfully executed in the current session.
        # Reset by the orchestrator at the start of each new "execute" request.
        self.last_executed_user_index: int = 0

        # Define all tools
        self.get_available_services_tool = StructuredTool.from_function(
            func=self._get_available_services,
            name="get_available_services",
            description="Get a list of all available ROS2 services that can be added to the action sequence. Returns a JSON list of services with their types.",
            args_schema=GetSequenceSummaryInput  # Reuse empty schema
        )

        self.add_service_to_sequence_tool = StructuredTool.from_function(
            func=self._add_service_to_sequence_structured,
            name="add_service_to_sequence",
            description="""Add a ROS2 service to the action sequence at a specific index. 
            Specify service_client (required), index (required), and optionally service_type and service_name.
            Example: service_client="/move_robot", index=1, service_name="Move to Position 1"
            Returns success or failure message.""",
            args_schema=AddServiceToSequenceInput
        )

        self.add_user_interaction_tool = StructuredTool.from_function(
            func=self._add_user_interaction_structured,
            name="add_user_interaction",
            description="""Add a user interaction action to the sequence at a specific index. This pauses execution and waits for user confirmation.
            Specify index (required), action_name (required), action_description (required), and optionally interaction_mode ('terminal' or 'gui').
            Example: index=2, action_name="Confirm Position", action_description="Please confirm the robot is in the correct position"
            Returns success or failure message.""",
            args_schema=AddUserInteractionInput
        )

        self.set_action_parameters_tool = StructuredTool.from_function(
            func=self._set_action_parameters_structured,
            name="set_action_parameters",
            description="""Set or update parameters for an action at a specific index in the sequence.
            For nested messages (like Vector3, Quaternion), use nested dictionaries with the field names.
            For arrays of nested messages, use a list of dictionaries (NOT a list of strings).
            Example for simple params: index=1, parameters={"speed": 0.5, "timeout": 10.0}
            Example for nested params: index=3, parameters={"translation": {"x": 0.1, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "execute_movement": true}
            Example for array of nested messages (e.g. DispensePoint[]): index=5, parameters={"dispense_points": [{"frame_name": "Point_1", "time_ms": 500.0, "dispense_z_offset_mm": 0.0}, {"frame_name": "Point_2", "time_ms": 500.0, "dispense_z_offset_mm": 0.0}]}
            WRONG: parameters={"dispense_points": ["Point_1", "Point_2"]}  ← bare strings are never valid for message arrays
            Returns success or failure message.""",
            args_schema=SetActionParametersInput
        )

        self.get_action_list_tool = Tool(
            name="get_action_list",
            func=self._get_action_list,
            description="""[EXPENSIVE - USE SPARINGLY] Get full details for ALL actions in sequence.
            COST: 5000-20000 tokens for large sequences.
            WHEN TO USE: Only when you absolutely need full type/client details for MULTIPLE actions.
            ALTERNATIVES: Use get_sequence_summary (overview) or get_action_at_index (specific queries) instead.
            99% of queries should use the more efficient tools.
            Returns: Complete list with index, name, type, client for every action."""
        )

        self.delete_action_tool = Tool(
            name="delete_action",
            func=self._delete_action,
            description="""Delete an action from the sequence at a specific index.
            Input should be a JSON string with key: 'index' (required).
            Example: {"index": 2}
            Returns success or failure message."""
        )

        self.move_action_tool = StructuredTool.from_function(
            func=self._move_action_structured,
            name="move_action",
            description="""Move action from old_index to new_index.
            
            CRITICAL: To place action A BEFORE action B at index N, use new_index = N-1
            (Action A is removed first, shifting B down, then A inserted before B's new position)
            
            Example: Move Dispense(16) BEFORE Grip(23) → use old_index=16, new_index=22
            Wrong: old_index=16, new_index=23 places Dispense AFTER Grip!
            
            VERIFICATION: After moving to new_index, verify the action is AT new_index (not new_index-1 or new_index+1).
            Check get_action_at_index(new_index) returns the moved action's name.""",
            args_schema=MoveActionInput
        )

        self.execute_sequence_tool = StructuredTool.from_function(
            func=self._execute_sequence_structured,
            name="execute_sequence",
            description="""Execute the complete action sequence starting from a specific index (default 0).
            Returns execution log as JSON.""",
            args_schema=ExecuteSequenceInput
        )

        self.execute_single_action_tool = Tool(
            name="execute_single_action",
            func=self._execute_single_action,
            description="""Execute a single action at a specific index without executing the entire sequence.
            CRITICAL: NEVER call multiple execute_single_action in parallel. The robot can only perform one action at a time and actions may depend on each other. Always execute sequentially (one per response).
            Input should be a JSON string with key: 'index' (required).
            Example: {"index": 1}
            Returns success or failure with execution details."""
        )

        self.clear_sequence_tool = Tool(
            name="clear_sequence",
            func=self._clear_sequence,
            description="Clear all actions from the current sequence. Use with caution as this cannot be undone. Returns success message."
        )

        self.save_sequence_tool = Tool(
            name="save_sequence",
            func=self._save_sequence,
            description="""Save the current action sequence to a file.
            Input should be a JSON string with key: 'file_name' (required).
            Example: {"file_name": "my_sequence.json"}
            Returns success or failure message."""
        )

        self.load_sequence_tool = Tool(
            name="load_sequence",
            func=self._load_sequence,
            description="""Load an action sequence from a file.
            Input should be a JSON string with key: 'file_name' (required).
            Example: {"file_name": "my_sequence.json"}
            Returns success or failure message."""
        )

        self.get_service_parameters_tool = Tool(
            name="get_service_parameters",
            func=self._get_service_parameters,
            description="""Get the request and response parameter structure for one or more ROS2 services.
            Input should be a JSON string with key: 'service_clients' (required, can be a single string or list of strings).
            Example: {"service_clients": "/move_robot"} or {"service_clients": ["/move_robot", "/get_position"]}
            Returns the service type and full parameter structure (request and response fields) for each service."""
        )

        self.get_parameter_value_recommendations_tool = StructuredTool.from_function(
            func=self._get_parameter_value_recommendations_structured,
            name="get_parameter_value_recommendations",
            description="""Get recommended values for action parameters based on parameter type and available system resources.
            Returns value sets like TF frames, vision cameras/processes, assembly components, etc.
            Optionally filter by parameter_type (e.g., 'string', 'str', 'uint32', 'double').
            If parameter_type is omitted, returns ALL available value sets.
            Example: parameter_type="string" or parameter_type="str"
            Use this to discover valid values for parameters like frame names, camera names, component names, etc.""",
            args_schema=GetParameterValueRecommendationsInput
        )

        self.get_action_at_index_tool = Tool(
            name="get_action_at_index",
            func=self._get_action_at_index,
            description="""PREFERRED for querying specific indices. Get details about one or more actions at specific indices.
            WHEN TO USE: Checking specific positions, verifying moves, confirming action names/types.
            EFFICIENCY: ~50 tokens per action vs ~5000 tokens for get_sequence_summary.
            Input formats:
              - Single: 28 or {"index": 28}
              - Multiple: {"indices": [22, 23, 41, 42]} or {"index": [22, 23, 41, 42]}
            Returns: name, type, client, and active status for requested action(s).
            IMPORTANT: Use this instead of get_sequence_summary when checking specific indices (even 10+ indices is more efficient)."""
        )

        self.get_sequence_summary_tool = StructuredTool.from_function(
            func=self._get_sequence_summary_structured,
            name="get_sequence_summary",
            description="""Get overview of entire sequence - indices, names, and active status only.
            WHEN TO USE: Initial planning, finding multiple items by name, showing user the full sequence.
            EFFICIENCY: ~500-5000 tokens depending on sequence length.
            WARNING: Do NOT call repeatedly after every move_action! Indices update automatically.
            STRATEGY: Call ONCE at start for planning, then use get_action_at_index to verify specific positions.
            Returns: lightweight list with index, name, active status (no parameters or full details).""",
            args_schema=GetSequenceSummaryInput
        )

        self.get_action_parameters_tool = Tool(
            name="get_action_parameters",
            func=self._get_action_parameters,
            description="""Get current parameter values for an action at a specific index.
            Returns the full request dictionary with all current parameter values.
            Input should be a JSON string with key: 'index' (required).
            Example: {"index": 27}
            Returns current parameter values as a dictionary."""
        )

        self.build_sequence_from_plan_tool = StructuredTool.from_function(
            func=self._build_sequence_from_plan,
            name="build_sequence_from_plan",
            description=(
                "[BATCH BUILDER] Add multiple actions to the sequence in a single call. "
                "PREFER this over calling add_service_to_sequence + set_action_parameters "
                "repeatedly when building a new sequence from scratch.\n\n"
                "Provide an ordered list of ActionSpec objects, each with:\n"
                "  - service_client (required): ROS2 service client name\n"
                "  - name (required): display name for the action\n"
                "  - parameters (optional): dict of parameter values to set\n"
                "  - service_type (optional): service type string\n\n"
                "Set clear_existing=True to wipe the current sequence first.\n"
                "Returns a per-action result report (success/failure for each action)."
            ),
            args_schema=BuildSequenceFromPlanInput,
        )

        self.load_and_modify_sequence_tool = StructuredTool.from_function(
            func=self._load_and_modify_sequence,
            name="load_and_modify_sequence",
            description=(
                "Load an existing .rsap.json sequence file into the RSAP instance. "
                "This replaces the current sequence with the loaded one. "
                "After loading, use the standard atomic tools (set_action_parameters, "
                "delete_action, move_action, add_service_to_sequence) to adapt it. "
                "Input: absolute file path to a .rsap.json file. "
                "Use list_available_rsap_sequences to discover available files."
            ),
            args_schema=LoadAndModifySequenceInput,
        )

    def _get_available_services(self, input_str: str = "") -> str:
        """Get list of available ROS2 services."""
        try:
            self.rsap.initialize_service_list()
            services = self.rsap.get_active_services()
            filtered_services = self.rsap.get_active_client_whtlist()
            try:
                blacklist_path = get_package_share_directory('pm_co_pilot_planning') + '/blacklist.yaml'
                with open(blacklist_path, 'r') as f:
                    blacklist = yaml.safe_load(f)
                blacklisted_names = set(blacklist.get('clients_by_name', []))
                blacklisted_types = set(blacklist.get('clients_by_type', []))
                result = [{"client": svc[0], "type": svc[1][0]} for svc in services
                          if svc[0] in filtered_services
                          and svc[0] not in blacklisted_names
                          and svc[1][0] not in blacklisted_types]
            except Exception:
                result = [{"client": svc[0], "type": svc[1][0]} for svc in services if svc[0] in filtered_services]
            return json.dumps({"services": result, "count": len(result)})
        except Exception as e:
            return ToolResponse.error(str(e))

    def _add_service_to_sequence_structured(self, service_client: str, index: int,
                                           service_type: Optional[str] = None,
                                           service_name: Optional[str] = None) -> str:
        """Add a service to the action sequence (StructuredTool version).
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal).
        Uses lock to prevent parallel execution that causes wrong ordering.
        Clamps index to valid range when called in parallel."""

        with self._sequence_lock:
            try:
                if not service_client:
                    return ToolResponse.error("service_client is required")

                internal_index = index - 1
                current_length = len(self.rsap.action_list)
                original_index = index
                if internal_index > current_length:
                    internal_index = current_length
                    index = internal_index + 1
                elif internal_index < 0:
                    internal_index = 0
                    index = 1

                success = self.rsap.append_service_to_action_list_at_index(
                    service_client=service_client,
                    index=internal_index,
                    service_type=service_type,
                    service_name=service_name
                )

                if not success:
                    return ToolResponse.error("Failed to add service")

                result = {
                    "success": True,
                    "message": f"Service '{service_client}' added at position {index}",
                    "current_sequence_length": len(self.rsap.action_list)
                }
                if original_index != index:
                    result["note"] = f"Requested index {original_index} was clamped to {index} (end of sequence)"
                return json.dumps(result)

            except Exception as e:
                return ToolResponse.error(str(e))

    def _add_user_interaction_structured(self, index: int, action_name: str,
                                        action_description: str,
                                        interaction_mode: str = "terminal") -> str:
        """Add a user interaction to the action sequence (StructuredTool version).
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            if not action_name or not action_description:
                return ToolResponse.error("action_name and action_description are required")

            internal_index = index - 1

            from ros_sequential_action_programmer.submodules.action_classes.UserInteractionAction import TERMINAL, GUI
            mode = GUI if interaction_mode.lower() == "gui" else TERMINAL

            success = self.rsap.append_user_interaction_to_action_list_at_index(
                index=internal_index,
                action_name=action_name,
                action_description=action_description,
                interaction_mode=mode
            )

            if success:
                return ToolResponse.success(
                    message=f"User interaction '{action_name}' added at position {index}",
                    current_sequence_length=len(self.rsap.action_list),
                )
            else:
                return ToolResponse.error("Failed to add user interaction")

        except Exception as e:
            return ToolResponse.error(str(e))

    def _set_action_parameters_structured(self, index: int, parameters: Dict[str, Any]) -> str:
        """Set parameters for an action at a specific index (StructuredTool version).
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            if index is None:
                return ToolResponse.error("index is required")

            if not parameters or not isinstance(parameters, dict):
                return ToolResponse.error(
                    "'parameters' dict is required. You must provide the parameter values to set.",
                    expected_format={"index": "<int>", "parameters": {"param_name": "value", "...": "..."}},
                    hint="Use get_service_parameters or get_action_parameters to discover the parameter names first.",
                )

            internal_index, err = to_internal_index(index, len(self.rsap.action_list))
            if err:
                return ToolResponse.error(
                    f"{err}. Action must be added before setting parameters.",
                    hint="When calling add_service and set_parameters in parallel, the action may not exist yet. Call them sequentially instead.",
                )

            action = self.rsap.get_action_at_index(internal_index)

            if not hasattr(action, 'request'):
                return ToolResponse.error(
                    f"Action at index {index} does not support parameter setting (type: {type(action).__name__})"
                )

            if not hasattr(action, 'get_request_as_ordered_dict'):
                return ToolResponse.error(
                    f"Action at index {index} does not support parameter retrieval (type: {type(action).__name__})"
                )

            current_params = action.get_request_as_ordered_dict()
            merged_params = self._deep_merge(dict(current_params), parameters)
            params_copy = copy.deepcopy(merged_params)

            try:
                set_message_fields(action.request, params_copy)
            except Exception as e:
                return ToolResponse.error(
                    f"Failed to set parameters: {str(e)}. Make sure parameter names and types match the service definition."
                )

            return ToolResponse.success(message=f"Parameters set for action at position {index}")

        except Exception as e:
            return ToolResponse.error(str(e))

    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with update values taking precedence.
        Handles lists by merging elements at the same index."""
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    # Recursively merge nested dictionaries
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(value, list) and isinstance(result[key], list):
                    # Merge lists element by element
                    merged_list = []
                    for i in range(max(len(result[key]), len(value))):
                        if i < len(value) and i < len(result[key]):
                            # Both have element at index i
                            if isinstance(value[i], dict) and isinstance(result[key][i], dict):
                                merged_list.append(self._deep_merge(result[key][i], value[i]))
                            else:
                                merged_list.append(value[i])
                        elif i < len(value):
                            # Only update has element at index i
                            merged_list.append(value[i])
                        else:
                            # Only base has element at index i
                            merged_list.append(result[key][i])
                    result[key] = merged_list
                else:
                    # Simple value, just replace
                    result[key] = value
            else:
                # New key, add it
                result[key] = value
        
        return result

    def _get_action_list(self, input_str: str = "") -> str:
        """Get the current action sequence."""
        try:
            actions = [ActionHelper.to_dict(action, idx) for idx, action in enumerate(self.rsap.action_list)]
            return json.dumps({
                "actions": actions,
                "total_count": len(actions),
                "current_index": self.rsap.get_current_action_index()
            })
        except Exception as e:
            return ToolResponse.error(str(e))

    def _get_action_parameters(self, input_str: str) -> str:
        """Get current parameter values for an action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index, err = parse_tool_index(input_str)
            if err:
                return ToolResponse.error(err)

            internal_index, err = to_internal_index(user_index, len(self.rsap.action_list))
            if err:
                return ToolResponse.error(err)

            action = self.rsap.get_action_at_index(internal_index)

            if not hasattr(action, 'get_request_as_ordered_dict'):
                return ToolResponse.error(
                    f"Action at index {user_index} does not have parameters (type: {type(action).__name__})"
                )

            current_params = action.get_request_as_ordered_dict()
            return ToolResponse.success(index=user_index, parameters=dict(current_params))

        except Exception as e:
            return ToolResponse.error(str(e))

    def _delete_action(self, input_str: str) -> str:
        """Delete an action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index, err = parse_tool_index(input_str)
            if err:
                return ToolResponse.error(err)

            internal_index, err = to_internal_index(user_index, len(self.rsap.action_list))
            if err:
                return ToolResponse.error(err)

            success = self.rsap.delete_action_at_index(internal_index)

            if success:
                return ToolResponse.success(
                    message=f"Action at position {user_index} deleted",
                    remaining_actions=len(self.rsap.action_list),
                )
            else:
                return ToolResponse.error(f"Failed to delete action at position {user_index}")

        except Exception as e:
            return ToolResponse.error(str(e))

    def _move_action_structured(self, old_index: int, new_index: int) -> str:
        """Move an action from one index to another (StructuredTool version).
        Note: Accepts 1-based indices (GUI) and converts to 0-based (internal)."""
        try:
            from_user = int(old_index)
            to_user = int(new_index)
            action_count = len(self.rsap.action_list)

            from_internal, err = to_internal_index(from_user, action_count)
            if err:
                return ToolResponse.error(f"old_index: {err}")

            to_internal, err = to_internal_index(to_user, action_count)
            if err:
                return ToolResponse.error(f"new_index: {err}")

            success = self.rsap.move_action_at_index_to_index(from_internal, to_internal)

            if success:
                return ToolResponse.success(
                    message=f"Action moved from position {from_user} to position {to_user}"
                )
            else:
                return ToolResponse.error("Failed to move action")

        except Exception as e:
            return ToolResponse.error(str(e))


    def _execute_sequence_structured(self, start_index: int = 0) -> str:
        """Core implementation for execute_sequence."""
        try:
            internal_start = max(0, start_index - 1) if start_index > 0 else 0

            success, final_index = self.rsap.execute_action_list(internal_start)

            result = {
                "success": success,
                "start_index": start_index,
                "final_index": final_index + 1 if final_index is not None else None,
                "message": "Sequence executed successfully" if success else f"Sequence execution failed at index {final_index + 1}"
            }

            if not success and final_index is not None:
                failed_action = self.rsap.get_action_at_index(final_index)
                if failed_action:
                    result["failed_action_name"] = failed_action.get_name() if hasattr(failed_action, 'get_name') else str(failed_action)
                    result.update(ActionHelper.extract_srv_response(failed_action))

            return json.dumps(result)

        except Exception as e:
            return ToolResponse.error(str(e))

    @staticmethod
    def _compute_state_diff(before: dict, after: dict) -> list:
        """Compute a human-readable list of scene state changes between two snapshots."""
        changes = []
        for name in after:
            if name not in before:
                changes.append(f"{name} appeared in scene")
        for name in before:
            if name not in after:
                changes.append(f"{name} removed from scene")
        for name in after:
            if name not in before:
                continue
            # Object-level property changes
            for prop in ("is_gripped", "is_assembled", "parent_frame"):
                if before[name].get(prop) != after[name].get(prop):
                    changes.append(
                        f"{name}.{prop} changed to {after[name][prop]}"
                    )
            # Frame property changes
            before_frames = {f["frame_name"]: f for f in before[name].get("frames", [])}
            after_frames = {f["frame_name"]: f for f in after[name].get("frames", [])}
            for fname in after_frames:
                if fname not in before_frames:
                    changes.append(f"{name}: frame {fname} added")
                    continue
                before_props = before_frames[fname].get("properties", {})
                after_props = after_frames[fname].get("properties", {})
                if before_props != after_props:
                    for prop_type, prop_values in after_props.items():
                        old_values = before_props.get(prop_type, {})
                        for key, val in prop_values.items():
                            if old_values.get(key) != val:
                                changes.append(
                                    f"{fname}.{prop_type}.{key} changed to {val}"
                                )
            for fname in before_frames:
                if fname not in after_frames:
                    changes.append(f"{name}: frame {fname} removed")
        return changes

    def _execute_single_action(self, input_str: str) -> str:
        """Execute a single action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index, err = parse_tool_index(input_str)
            if err:
                return ToolResponse.error(err)

            internal_index, err = to_internal_index(user_index, len(self.rsap.action_list))
            if err:
                return ToolResponse.error(err)

            # Snapshot scene state before execution
            before_snapshot = {}
            if self._assembly_knowledge is not None:
                self._assembly_knowledge._ensure_scene_updated()
                before_snapshot = self._assembly_knowledge._get_scene_snapshot()

            self.rsap.set_current_action(internal_index)
            success = self.rsap.execute_current_action()

            if success:
                self.last_executed_user_index = max(self.last_executed_user_index, user_index)

            result = {
                "success": success,
                "index": user_index,
                "message": ActionHelper.extract_srv_response(self.rsap.get_action_at_index((internal_index)))
            }

            # Snapshot after and report state changes
            if self._assembly_knowledge is not None:
                self._assembly_knowledge._ensure_scene_updated()
                after_snapshot = self._assembly_knowledge._get_scene_snapshot()
                state_changes = self._compute_state_diff(before_snapshot, after_snapshot)
                if state_changes:
                    result["state_changes"] = state_changes

            # result["current_scene_state"] = self._assembly_knowledge._get_scene_snapshot()

            if not success:
                action = self.rsap.get_action_at_index(internal_index)
                result.update(ActionHelper.extract_srv_response(action))

            return json.dumps(result)

        except Exception as e:
            return ToolResponse.error(str(e))

    def _clear_sequence(self, input_str: str = "") -> str:
        """Clear all actions from the sequence."""
        try:
            count = len(self.rsap.action_list)
            self.rsap.action_list.clear()
            self.rsap.current_action_index = 0
            return ToolResponse.success(
                message=f"Cleared {count} actions from sequence",
                remaining_actions=len(self.rsap.action_list),
            )
        except Exception as e:
            return ToolResponse.error(str(e))

    def _save_sequence(self, input_str: str) -> str:
        """Save the current sequence to a file."""
        try:
            params = json.loads(input_str)
            file_name = params.get("file_name")

            if not file_name:
                return ToolResponse.error("file_name is required")

            self.rsap.rsap_file_manager.set_folder_path("/home/match-pm/Desktop")
            self.rsap.rsap_file_manager.set_sequence_name(file_name)
            success = self.rsap.rsap_file_manager.save_to_JSON()

            if success:
                return ToolResponse.success(
                    message=f"Sequence saved to {file_name}",
                    action_count=len(self.rsap.action_list),
                )
            else:
                return ToolResponse.error("Failed to save sequence")

        except json.JSONDecodeError as e:
            return ToolResponse.error(f"Invalid JSON input: {str(e)}")
        except Exception as e:
            return ToolResponse.error(str(e))

    def _load_sequence(self, input_str: str) -> str:
        """Load a sequence from a file."""
        try:
            params = json.loads(input_str)
            file_path = params.get("file_path")

            if not file_path:
                return ToolResponse.error("file_path is required")

            success = self.rsap.rsap_file_manager.load_from_JSON(file_path)

            if success:
                return ToolResponse.success(
                    message=f"Sequence loaded from {file_path}",
                    action_count=len(self.rsap.action_list),
                )
            else:
                return ToolResponse.error("Failed to load sequence")

        except json.JSONDecodeError as e:
            return ToolResponse.error(f"Invalid JSON input: {str(e)}")
        except Exception as e:
            return ToolResponse.error(str(e))

    def _get_parameter_value_recommendations_structured(self, parameter_type: Optional[str] = None) -> str:
        """Get recommended parameter values based on type and available system resources."""
        try:
            value_set_generator, err = ValueSetHelper.get_generator(self.rsap)
            if err:
                return err

            if hasattr(value_set_generator, 'update'):
                value_set_generator.update()

            if parameter_type:
                value_set_names = value_set_generator.value_sets.get_all_value_set_names(parameter_type)
            else:
                value_set_names = value_set_generator.value_sets.get_all_value_set_names()

            excluded_sets = {'instructions_in_scene', 'vision_cameras', 'vision_processes',
                           'test_set_1', 'test_set_2', 'test_set_3', 'test_set_4'}
            value_set_names = [name for name in value_set_names if name not in excluded_sets]

            recommendations = {}
            for set_name in value_set_names:
                try:
                    value_set = value_set_generator.value_sets.get_set_for_set_name(set_name)
                    recommendations[set_name] = {
                        "type": value_set.value_set_type,
                        "values": value_set.get_values_list()
                    }
                except Exception as e:
                    self.service_node.get_logger().warn(f"Could not get values for set '{set_name}': {e}")

            return ToolResponse.success(
                parameter_type=parameter_type if parameter_type else "all",
                value_sets=recommendations,
                count=len(recommendations),
            )

        except Exception as e:
            return ToolResponse.error(str(e))
    

    def _get_service_parameters(self, input_str) -> str:
        """Get parameter structure for one or more services."""
        try:
            # Handle case where LLM passes a dict or list directly instead of a JSON string
            if isinstance(input_str, dict):
                service_clients = input_str.get("service_clients", input_str.get("__arg1"))
                if service_clients is None:
                    service_clients = list(input_str.values())[0] if input_str else None
            elif isinstance(input_str, list):
                service_clients = input_str
            elif isinstance(input_str, str):
                try:
                    params = json.loads(input_str)
                    if isinstance(params, dict):
                        service_clients = params.get("service_clients", params.get("__arg1"))
                    elif isinstance(params, list):
                        service_clients = params
                    else:
                        service_clients = input_str
                except (json.JSONDecodeError, AttributeError):
                    service_clients = input_str
            else:
                service_clients = str(input_str)

            if not service_clients:
                return ToolResponse.error("service_clients is required")

            if isinstance(service_clients, str):
                service_clients = [service_clients]

            if not isinstance(service_clients, list):
                return ToolResponse.error("service_clients must be a string or list of strings")

            self.rsap.initialize_service_list()
            services = self.rsap.get_active_services()
            filtered_services = set(self.rsap.get_active_client_whtlist())
            try:
                blacklist_path = get_package_share_directory('pm_co_pilot_planning') + '/blacklist.yaml'
                with open(blacklist_path, 'r') as f:
                    blacklist = yaml.safe_load(f)
                blacklisted_names = set(blacklist.get('clients_by_name', []))
                blacklisted_types = set(blacklist.get('clients_by_type', []))
            except Exception:
                blacklisted_names = set()
                blacklisted_types = set()
            available_clients = [svc[0] for svc in services
                                 if svc[0] in filtered_services
                                 and svc[0] not in blacklisted_names
                                 and svc[1][0] not in blacklisted_types]

            missing_services = [svc for svc in service_clients if svc not in available_clients]
            if missing_services:
                all_active_clients = [svc[0] for svc in services]
                in_system_but_filtered = [svc for svc in missing_services if svc in all_active_clients]

                error_msg = f"Service(s) not found in available services: {', '.join(missing_services)}."
                if in_system_but_filtered:
                    error_msg += f" Note: {', '.join(in_system_but_filtered)} exist but may be filtered by whitelist/blacklist."
                else:
                    error_msg += " Make sure the service(s) are running."

                return ToolResponse.error(
                    error_msg,
                    requested=service_clients,
                    missing=missing_services,
                    available_count=len(available_clients),
                )

            service_params = self.rsap.get_all_service_req_res_dict(service_clients)

            if not service_params:
                return ToolResponse.error(
                    "Could not retrieve parameter information. The service type may not be available or there was an error parsing the service definition.",
                    requested=service_clients,
                )

            return ToolResponse.success(services=service_params, count=len(service_params))

        except Exception as e:
            return ToolResponse.error(str(e))

    def _get_action_at_index(self, input_str: str) -> str:
        """Get details about one or more actions - much more token-efficient than getting full list.
        Note: Accepts 1-based indices (GUI) and converts to 0-based (internal)."""
        try:
            user_indices, err = parse_tool_indices(input_str)
            if err:
                return ToolResponse.error(err)

            results = []
            errors = []
            action_count = len(self.rsap.action_list)

            for user_index in user_indices:
                try:
                    user_index = int(user_index)
                    internal_index, err = to_internal_index(user_index, action_count)
                    if err:
                        errors.append(err)
                        continue

                    action = self.rsap.get_action_at_index(internal_index)
                    results.append(ActionHelper.to_dict(action, user_index))

                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid index {user_index}: {str(e)}")

            if len(results) == 1 and len(errors) == 0:
                return json.dumps({"success": True, **results[0]})
            else:
                response = {"success": len(results) > 0, "actions": results}
                if errors:
                    response["errors"] = errors
                return json.dumps(response)

        except Exception as e:
            return ToolResponse.error(str(e))

    def _get_sequence_summary_structured(self) -> str:
        """Get a lightweight summary of the sequence - just names and indices.
        Note: Returns 1-based indices matching GUI display."""
        try:
            actions = []
            for idx, action in enumerate(self.rsap.action_list):
                info = ActionHelper.to_dict(action, idx + 1)
                # Use 'active' key for backward compat in summary (not 'is_active')
                actions.append({
                    "index": info["index"],
                    "name": info["name"],
                    "active": info["is_active"],
                })

            return json.dumps({
                "total_count": len(self.rsap.action_list),
                "current_index": self.rsap.get_current_action_index() + 1,
                "actions": actions,
            })

        except Exception as e:
            return ToolResponse.error(str(e))

    def _build_sequence_from_plan(
        self,
        actions: List[ActionSpec],
        clear_existing: bool = False,
        start_index: Optional[int] = None,
    ) -> str:
        """
        Batch-create multiple sequence actions in one call.

        Each ActionSpec is added with add_service + set_parameters in a single
        locked operation so parallel callers cannot interleave.
        """
        try:
            if clear_existing:
                self.rsap.action_list.clear()
                if hasattr(self.rsap, 'set_current_action_index'):
                    self.rsap.set_current_action_index(0)
                elif hasattr(self.rsap, 'current_action_index'):
                    self.rsap.current_action_index = 0

            results = []
            errors = []

            for i, action_spec in enumerate(actions):
                with self._sequence_lock:
                    try:
                        # Determine insertion index
                        if start_index is not None:
                            insert_idx_0 = (start_index - 1) + i  # 0-based
                        else:
                            insert_idx_0 = len(self.rsap.action_list)  # append

                        # Clamp to valid range
                        insert_idx_0 = max(0, min(insert_idx_0, len(self.rsap.action_list)))

                        success = self.rsap.append_service_to_action_list_at_index(
                            service_client=action_spec.service_client,
                            index=insert_idx_0,
                            service_type=action_spec.service_type,
                            service_name=action_spec.name,
                        )

                        if not success:
                            errors.append({
                                "action_index": i + 1,
                                "name": action_spec.name,
                                "error": "append_service_to_action_list_at_index returned False",
                            })
                            results.append({"action_index": i + 1, "name": action_spec.name, "success": False})
                            continue

                        # Set parameters if provided
                        if action_spec.parameters:
                            try:
                                rsap_action = self.rsap.get_action_at_index(insert_idx_0)
                                current_params = dict(rsap_action.get_request_as_ordered_dict())
                                merged = self._deep_merge(current_params, action_spec.parameters)
                                merged_copy = copy.deepcopy(merged)
                                set_message_fields(rsap_action.request, merged_copy)
                            except Exception as param_error:
                                errors.append({
                                    "action_index": i + 1,
                                    "name": action_spec.name,
                                    "error": f"Parameter setting failed: {param_error}",
                                })
                                results.append({
                                    "action_index": i + 1,
                                    "name": action_spec.name,
                                    "success": False,
                                    "note": "Action added but parameters could not be set",
                                })
                                continue

                        results.append({
                            "action_index": i + 1,
                            "name": action_spec.name,
                            "inserted_at": insert_idx_0 + 1,  # 1-based for display
                            "success": True,
                        })

                    except Exception as e:
                        errors.append({"action_index": i + 1, "name": action_spec.name, "error": str(e)})
                        results.append({"action_index": i + 1, "name": action_spec.name, "success": False})

            successful = sum(1 for r in results if r.get("success"))
            return json.dumps({
                "success": successful == len(actions),
                "total_actions": len(actions),
                "successful": successful,
                "failed": len(actions) - successful,
                "sequence_length_after": len(self.rsap.action_list),
                "results": results,
                "errors": errors if errors else None,
            })

        except Exception as e:
            return ToolResponse.error(str(e))

    def _load_and_modify_sequence(self, file_path: str) -> str:
        """Load an RSAP sequence from a .rsap.json file."""
        try:
            if not file_path:
                return ToolResponse.error("file_path is required")

            self.rsap.rsap_file_manager.load_from_JSON(file_path)
            count = len(self.rsap.action_list)
            return ToolResponse.success(
                message=f"Loaded sequence with {count} actions from '{file_path}'",
                action_count=count,
            )
        except Exception as e:
            return ToolResponse.error(str(e))


