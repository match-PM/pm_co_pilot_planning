from langchain_core.tools import Tool, tool, StructuredTool
from pydantic import BaseModel, Field
from rclpy.node import Node

import json
import copy
import threading
from typing import Optional, Dict, Any, List

from ament_index_python.packages import get_package_share_directory
from ros_sequential_action_programmer.submodules.RosSequentialActionProgrammer import RosSequentialActionProgrammer
from rosidl_runtime_py.set_message import set_message_fields


# Pydantic schemas for StructuredTool inputs
class SetActionParametersInput(BaseModel):
    """Input schema for set_action_parameters tool."""
    index: int = Field(description="1-based index of the action in the sequence (GUI index)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Dictionary of parameter key-value pairs to set. For nested messages like Vector3 or Quaternion, use nested dictionaries.")


class MoveActionInput(BaseModel):
    """Input schema for move_action tool."""
    old_index: int = Field(description="Current 1-based index of the action to move (GUI index)")
    new_index: int = Field(description="Target 1-based index where the action should be moved (GUI index)")


class GetSequenceSummaryInput(BaseModel):
    """Input schema for get_sequence_summary tool (no parameters required)."""
    pass


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
        description="Parameter key-value pairs to set on the action. Nested dicts for nested ROS messages."
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


class Tools:
    """
    The Tools class provides a set of tools that can be used by the agent.
    Each tool is a function that performs a specific action to control the RosSequentialActionProgrammer.
    """
    def __init__(self, service_node: Node, rsap_instance=None):
        self.service_node = service_node
        # Use provided RSAP instance or create a new one
        if rsap_instance:
            self.rsap = rsap_instance
        else:
            self.rsap = RosSequentialActionProgrammer(service_node)
        
        # Add lock for sequence modification operations
        self._sequence_lock = threading.Lock()

        # Define all tools
        self.get_available_services_tool = StructuredTool.from_function(
            func=self._get_available_services,
            name="get_available_services",
            description="Get a list of all available ROS2 services that can be added to the action sequence. Returns a JSON list of services with their types.",
            args_schema=GetSequenceSummaryInput  # Reuse empty schema
        )

        # self.get_available_ros_actions_tool = Tool(
        #     name="get_available_ros_actions",
        #     func=self._get_available_ros_actions,
        #     description="Get a list of all available ROS2 actions that can be added to the action sequence. Returns a JSON list of actions with their types."
        # )

        self.add_service_to_sequence_tool = StructuredTool.from_function(
            func=self._add_service_to_sequence_structured,
            name="add_service_to_sequence",
            description="""Add a ROS2 service to the action sequence at a specific index. 
            Specify service_client (required), index (required), and optionally service_type and service_name.
            Example: service_client="/move_robot", index=1, service_name="Move to Position 1"
            Returns success or failure message.""",
            args_schema=AddServiceToSequenceInput
        )

        # self.add_ros_action_to_sequence_tool = StructuredTool.from_function(
        #     func=self._add_ros_action_to_sequence_structured,
        #     name="add_ros_action_to_sequence",
        #     description="""Add a ROS2 action to the action sequence at a specific index.
        #     Specify action_client (required), index (required), and optionally action_type and action_name.
        #     Example: action_client="/navigate_to_pose", index=1, action_name="Navigate Home"
        #     Returns success or failure message.""",
        #     args_schema=AddRosActionToSequenceInput
        # )

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
            Example for simple params: index=1, parameters={"speed": 0.5, "timeout": 10.0}
            Example for nested params: index=3, parameters={"translation": {"x": 0.1, "y": 0.0, "z": 0.0}, "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}, "execute_movement": true}
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

        self.execute_sequence_tool = Tool(
            name="execute_sequence",
            func=self._execute_sequence,
            description="""Execute the complete action sequence starting from a specific index (default 0).
            Input should be a JSON string with optional key: 'start_index' (default 0).
            Example: {"start_index": 0} or just "{}" to start from beginning.
            Returns execution log as JSON."""
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
            # Filter by whitelist if available
            # filtered_services = self.rsap.get_active_client_blklist()
            filtered_services = self.rsap.get_active_client_whtlist()
            result = [{"client": svc[0], "type": svc[1][0]} for svc in services if svc[0] in filtered_services]
            return json.dumps({"services": result, "count": len(result)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # def _get_available_ros_actions(self, input_str: str = "") -> str:
    #     """Get list of available ROS2 actions."""
    #     try:
    #         self.rsap.initialize_ros_action_list()
    #         actions = self.rsap.get_active_ros_actions()
    #         result = [{"client": act[0], "type": act[1][0]} for act in actions]
    #         return json.dumps({"actions": result, "count": len(result)})
    #     except Exception as e:
    #         return json.dumps({"error": str(e)})

    def _add_service_to_sequence_structured(self, service_client: str, index: int, 
                                           service_type: Optional[str] = None, 
                                           service_name: Optional[str] = None) -> str:
        """Add a service to the action sequence (StructuredTool version).
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal).
        Uses lock to prevent parallel execution that causes wrong ordering.
        Clamps index to valid range when called in parallel."""
        
        # Acquire lock to serialize add operations - prevents race conditions
        with self._sequence_lock:
            try:
                if not service_client:
                    return json.dumps({"success": False, "error": "service_client is required"})

                # Convert from 1-based (GUI) to 0-based (internal)
                internal_index = index - 1
                
                # Clamp index to valid range (handles parallel calls where requested index
                # may exceed current sequence length because earlier adds haven't happened yet)
                current_length = len(self.rsap.action_list)
                original_index = index
                if internal_index > current_length:
                    internal_index = current_length  # Append at end
                    index = internal_index + 1  # Update GUI index
                elif internal_index < 0:
                    internal_index = 0
                    index = 1

                success = self.rsap.append_service_to_action_list_at_index(
                    service_client=service_client,
                    index=internal_index,
                    service_type=service_type,
                    service_name=service_name
                )

                if success:
                    result = {
                        "success": True,
                        "message": f"Service '{service_client}' added at position {index}",
                        "current_sequence_length": len(self.rsap.action_list)
                    }
                    if original_index != index:
                        result["note"] = f"Requested index {original_index} was clamped to {index} (end of sequence)"
                    return json.dumps(result)
                else:
                    return json.dumps({"success": False, "error": "Failed to add service"})

            except Exception as e:
                return json.dumps({"success": False, "error": str(e)})

    def _add_service_to_sequence(self, input_str: str) -> str:
        """Add a service to the action sequence.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                params = input_str
            elif isinstance(input_str, str):
                params = json.loads(input_str)
            else:
                return json.dumps({"success": False, "error": "Invalid input format"})
            
            service_client = params.get("service_client")
            user_index = params.get("index", len(self.rsap.action_list) + 1)  # 1-based, default to end
            service_type = params.get("service_type")
            service_name = params.get("service_name")

            if not service_client:
                return json.dumps({"success": False, "error": "service_client is required"})

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = user_index - 1

            success = self.rsap.append_service_to_action_list_at_index(
                service_client=service_client,
                index=internal_index,
                service_type=service_type,
                service_name=service_name
            )

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"Service '{service_client}' added at position {user_index}",
                    "current_sequence_length": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": "Failed to add service"})

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    # def _add_ros_action_to_sequence_structured(self, action_client: str, index: int,
    #                                            action_type: Optional[str] = None,
    #                                            action_name: Optional[str] = None) -> str:
    #     """Add a ROS action to the action sequence (StructuredTool version).
    #     Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
    #     try:
    #         if not action_client:
    #             return json.dumps({"success": False, "error": "action_client is required"})
    #
    #         # Auto-discover action type if not provided
    #         if not action_type:
    #             self.rsap.initialize_ros_action_list()
    #             for action_tuple in self.rsap.list_of_active_ros_actions:
    #                 if action_tuple[0] == action_client:
    #                     action_type = action_tuple[1][0]
    #                     self.service_node.get_logger().info(f"Auto-discovered action type: {action_type} for client: {action_client}")
    #                     break
    #         
    #             if not action_type:
    #                 return json.dumps({
    #                     "success": False,
    #                     "error": f"Action client '{action_client}' not found. Use get_available_ros_actions to see available actions."
    #                 })
    #
    #         # Convert from 1-based (GUI) to 0-based (internal)
    #         internal_index = index - 1
    #
    #         success = self.rsap.append_ros_action_to_action_list_at_index(
    #             action_client=action_client,
    #             index=internal_index,
    #             action_type=action_type,
    #             action_name=action_name
    #         )
    #
    #         if success:
    #             return json.dumps({
    #                 "success": True,
    #                 "message": f"Action '{action_client}' added at position {index}",
    #                 "current_sequence_length": len(self.rsap.action_list)
    #             })
    #         else:
    #             return json.dumps({"success": False, "error": "Failed to add action"})
    #
    #     except Exception as e:
    #         return json.dumps({"success": False, "error": str(e)})



    def _add_user_interaction_structured(self, index: int, action_name: str, 
                                        action_description: str, 
                                        interaction_mode: str = "terminal") -> str:
        """Add a user interaction to the action sequence (StructuredTool version).
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            if not action_name or not action_description:
                return json.dumps({"success": False, "error": "action_name and action_description are required"})

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = index - 1

            # Convert string mode to constant if needed
            from ros_sequential_action_programmer.submodules.action_classes.UserInteractionAction import TERMINAL, GUI
            mode = GUI if interaction_mode.lower() == "gui" else TERMINAL

            success = self.rsap.append_user_interaction_to_action_list_at_index(
                index=internal_index,
                action_name=action_name,
                action_description=action_description,
                interaction_mode=mode
            )

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"User interaction '{action_name}' added at position {index}",
                    "current_sequence_length": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": "Failed to add user interaction"})

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _add_user_interaction(self, input_str: str) -> str:
        """Add a user interaction to the action sequence.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                params = input_str
            elif isinstance(input_str, str):
                params = json.loads(input_str)
            else:
                return json.dumps({"success": False, "error": "Invalid input format"})
            
            user_index = params.get("index", len(self.rsap.action_list) + 1)  # 1-based, default to end
            action_name = params.get("action_name")
            action_description = params.get("action_description")
            interaction_mode = params.get("interaction_mode", "terminal")

            if not action_name or not action_description:
                return json.dumps({"success": False, "error": "action_name and action_description are required"})

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = user_index - 1

            # Convert string mode to constant if needed
            from ros_sequential_action_programmer.submodules.action_classes.UserInteractionAction import TERMINAL, GUI
            mode = GUI if interaction_mode.lower() == "gui" else TERMINAL

            success = self.rsap.append_user_interaction_to_action_list_at_index(
                index=internal_index,
                action_name=action_name,
                action_description=action_description,
                interaction_mode=mode
            )

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"User interaction '{action_name}' added at position {user_index}",
                    "current_sequence_length": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": "Failed to add user interaction"})

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


    def _set_action_parameters_structured(self, index: int, parameters: Dict[str, Any]) -> str:
        """Set parameters for an action at a specific index (StructuredTool version).
        Uses ActionBaseClass methods to properly handle nested structures and preserve existing values.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal).
        Returns helpful error if action doesn't exist yet (for parallel call debugging)."""
        try:
            if index is None:
                return json.dumps({"success": False, "error": "index is required"})
            
            if not parameters or not isinstance(parameters, dict):
                return json.dumps({
                    "success": False, 
                    "error": "'parameters' dict is required. You must provide the parameter values to set.",
                    "expected_format": {"index": "<int>", "parameters": {"param_name": "value", "...":  "..."}},
                    "hint": "Use get_service_parameters or get_action_parameters to discover the parameter names first."
                })

            # Check if index is valid BEFORE converting
            current_length = len(self.rsap.action_list)
            if index > current_length or index < 1:
                return json.dumps({
                    "success": False, 
                    "error": f"Index {index} out of range (valid: 1-{current_length}). Action must be added before setting parameters.",
                    "hint": "When calling add_service and set_parameters in parallel, the action may not exist yet. Call them sequentially instead."
                })

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = index - 1

            action = self.rsap.get_action_at_index(internal_index)
            
            # Check if action has request attribute (ServiceAction or RosAction)
            if not hasattr(action, 'request'):
                return json.dumps({
                    "success": False,
                    "error": f"Action at index {index} does not support parameter setting (type: {type(action).__name__})"
                })

            # Get current parameters to merge with new ones
            if hasattr(action, 'get_request_as_ordered_dict'):
                current_params = action.get_request_as_ordered_dict()
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Action at index {index} does not support parameter retrieval (type: {type(action).__name__})"
                })
            
            # Deep merge new parameters with current ones
            merged_params = self._deep_merge(dict(current_params), parameters)
            
            # Use set_message_fields directly - it handles nested structures properly
            # Make a deep copy to avoid modifying LangChain's message history
            params_copy = copy.deepcopy(merged_params)
            
            try:
                set_message_fields(action.request, params_copy)
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "error": f"Failed to set parameters: {str(e)}. Make sure parameter names and types match the service definition."
                })

            return json.dumps({
                "success": True,
                "message": f"Parameters set for action at position {index}"
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

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
            actions = []
            for idx, action in enumerate(self.rsap.action_list):
                action_info = {
                    "index": idx,
                    "name": action.get_name() if hasattr(action, 'get_name') else str(action),
                    "type": type(action).__name__,
                    "is_active": action.is_active() if hasattr(action, 'is_active') else True
                }
                
                # Add client info if it's a service or action
                if hasattr(action, 'client'):
                    action_info["client"] = action.client
                
                actions.append(action_info)

            return json.dumps({
                "actions": actions,
                "total_count": len(actions),
                "current_index": self.rsap.get_current_action_index()
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_action_parameters(self, input_str: str) -> str:
        """Get current parameter values for an action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index = None
            
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                user_index = input_str.get("index")
            elif isinstance(input_str, int):
                user_index = input_str
            elif isinstance(input_str, str):
                try:
                    params = json.loads(input_str)
                    user_index = params.get("index")
                except (json.JSONDecodeError, AttributeError):
                    try:
                        user_index = int(input_str)
                    except ValueError:
                        pass
            
            if user_index is None:
                return json.dumps({"success": False, "error": "index is required and must be a number"})
            
            # Ensure it's an integer
            user_index = int(user_index)
            
            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = user_index - 1
            
            if internal_index >= len(self.rsap.action_list) or internal_index < 0:
                return json.dumps({
                    "success": False,
                    "error": f"Index {user_index} out of range (valid: 1-{len(self.rsap.action_list)})"
                })
            
            action = self.rsap.get_action_at_index(internal_index)
            
            if not hasattr(action, 'get_request_as_ordered_dict'):
                return json.dumps({
                    "success": False,
                    "error": f"Action at index {user_index} does not have parameters (type: {type(action).__name__})"
                })
            
            current_params = action.get_request_as_ordered_dict()
            
            return json.dumps({
                "success": True,
                "index": user_index,
                "parameters": dict(current_params)
            })
            
        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _delete_action(self, input_str: str) -> str:
        """Delete an action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index = None
            
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                user_index = input_str.get("index")
            elif isinstance(input_str, int):
                user_index = input_str
            elif isinstance(input_str, str):
                try:
                    params = json.loads(input_str)
                    user_index = params.get("index")
                except (json.JSONDecodeError, AttributeError):
                    try:
                        user_index = int(input_str)
                    except ValueError:
                        pass

            if user_index is None:
                return json.dumps({"success": False, "error": "index is required and must be a number"})

            # Ensure it's an integer
            user_index = int(user_index)

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = user_index - 1

            if internal_index >= len(self.rsap.action_list) or internal_index < 0:
                return json.dumps({
                    "success": False,
                    "error": f"Index {user_index} out of range (valid: 1-{len(self.rsap.action_list)})"
                })

            success = self.rsap.delete_action_at_index(internal_index)

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"Action at position {user_index} deleted",
                    "remaining_actions": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": f"Failed to delete action at position {user_index}"})

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _move_action_structured(self, old_index: int, new_index: int) -> str:
        """Move an action from one index to another (StructuredTool version).
        Note: Accepts 1-based indices (GUI) and converts to 0-based (internal)."""
        try:
            from_user = int(old_index)
            to_user = int(new_index)

            if from_user is None or to_user is None:
                return json.dumps({"success": False, "error": "old_index and new_index are required"})

            # Convert from 1-based (GUI) to 0-based (internal)
            from_internal = from_user - 1
            to_internal = to_user - 1

            if from_internal >= len(self.rsap.action_list) or from_internal < 0:
                return json.dumps({
                    "success": False,
                    "error": f"old_index {from_user} out of range (valid: 1-{len(self.rsap.action_list)})"
                })

            if to_internal >= len(self.rsap.action_list) or to_internal < 0:
                return json.dumps({
                    "success": False,
                    "error": f"new_index {to_user} out of range (valid: 1-{len(self.rsap.action_list)})"
                })

            success = self.rsap.move_action_at_index_to_index(from_internal, to_internal)

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"Action moved from position {from_user} to position {to_user}"
                })
            else:
                return json.dumps({"success": False, "error": "Failed to move action"})

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _move_action(self, input_str: str) -> str:
        """Legacy move action method for backward compatibility."""
        try:
            if isinstance(input_str, dict):
                params = input_str
            elif isinstance(input_str, str):
                params = json.loads(input_str)
            else:
                return json.dumps({"success": False, "error": "Invalid input format"})
            
            return self._move_action_structured(
                old_index=params.get("old_index"),
                new_index=params.get("new_index")
            )
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _execute_sequence(self, input_str: str = "{}") -> str:
        """Execute the action sequence."""
        try:
            params = json.loads(input_str) if input_str and input_str != "{}" else {}
            start_index = params.get("start_index", 0)

            success, final_index = self.rsap.execute_action_list(start_index)
            
            result = {
                "success": success,
                "start_index": start_index,
                "final_index": final_index,
                "message": "Sequence executed successfully" if success else f"Sequence execution failed at index {final_index + 1}"
            }

            # Include error details from the failed action's log entry
            if not success and final_index is not None:
                failed_action = self.rsap.get_action_at_index(final_index)
                if failed_action:
                    result["failed_action_name"] = failed_action.get_name() if hasattr(failed_action, 'get_name') else str(failed_action)
                    log_entry = failed_action.get_log_entry() if hasattr(failed_action, 'get_log_entry') else {}
                    if log_entry:
                        if log_entry.get("message"):
                            result["error_message"] = log_entry["message"]
                        if log_entry.get("execution_time"):
                            result["execution_time"] = log_entry["execution_time"]
                    # Also check response_dict for error info
                    if hasattr(failed_action, 'response_dict') and failed_action.response_dict:
                        response = dict(failed_action.response_dict)
                        if "Error" in response:
                            result["error_detail"] = response["Error"]
                        else:
                            result["response"] = response

            return json.dumps(result)

        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _execute_single_action(self, input_str: str) -> str:
        """Execute a single action at a specific index.
        Note: Accepts 1-based index (GUI) and converts to 0-based (internal)."""
        try:
            user_index = None
            
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                user_index = input_str.get("index")
            elif isinstance(input_str, int):
                user_index = input_str
            elif isinstance(input_str, str):
                try:
                    params = json.loads(input_str)
                    user_index = params.get("index")
                except (json.JSONDecodeError, AttributeError):
                    try:
                        user_index = int(input_str)
                    except ValueError:
                        pass

            if user_index is None:
                return json.dumps({"success": False, "error": "index is required and must be a number"})

            # Ensure it's an integer
            user_index = int(user_index)

            # Convert from 1-based (GUI) to 0-based (internal)
            internal_index = user_index - 1

            if internal_index >= len(self.rsap.action_list) or internal_index < 0:
                return json.dumps({
                    "success": False, 
                    "error": f"Index {user_index} out of range (valid: 1-{len(self.rsap.action_list)})"
                })

            self.rsap.set_current_action(internal_index)
            success = self.rsap.execute_current_action()

            result = {
                "success": success,
                "index": user_index,
                "action_name": self.rsap.get_current_action_name(),
                "message": "Action executed successfully" if success else "Action execution failed"
            }

            # Include error details from the action's log entry on failure
            if not success:
                action = self.rsap.get_action_at_index(internal_index)
                log_entry = action.get_log_entry() if hasattr(action, 'get_log_entry') else {}
                if log_entry:
                    if log_entry.get("message"):
                        result["error_message"] = log_entry["message"]
                    if log_entry.get("execution_time"):
                        result["execution_time"] = log_entry["execution_time"]
                # Also check response_dict for error info (e.g. "Client not available")
                if hasattr(action, 'response_dict') and action.response_dict:
                    response = dict(action.response_dict)
                    if "Error" in response:
                        result["error_detail"] = response["Error"]
                    else:
                        result["response"] = response

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _clear_sequence(self, input_str: str = "") -> str:
        """Clear all actions from the sequence."""
        try:
            count = len(self.rsap.action_list)
            self.rsap.action_list.clear()
            self.rsap.current_action_index = 0

            return json.dumps({
                "success": True,
                "message": f"Cleared {count} actions from sequence",
                "remaining_actions": len(self.rsap.action_list)
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _save_sequence(self, input_str: str) -> str:
        """Save the current sequence to a file."""
        try:
            params = json.loads(input_str)
            file_name = params.get("file_name")

            if not file_name:
                return json.dumps({"success": False, "error": "file_name is required"})

            self.rsap.rsap_file_manager.set_folder_path("/home/match-pm/Desktop")
            self.rsap.rsap_file_manager.set_sequence_name(file_name)
            success = self.rsap.rsap_file_manager.save_to_JSON()

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"Sequence saved to {file_name}",
                    "action_count": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": "Failed to save sequence"})

        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _load_sequence(self, input_str: str) -> str:
        """Load a sequence from a file."""
        try:
            params = json.loads(input_str)
            file_path = params.get("file_path")

            if not file_path:
                return json.dumps({"success": False, "error": "file_path is required"})

            success = self.rsap.rsap_file_manager.load_from_JSON(file_path)

            if success:
                return json.dumps({
                    "success": True,
                    "message": f"Sequence loaded from {file_path}",
                    "action_count": len(self.rsap.action_list)
                })
            else:
                return json.dumps({"success": False, "error": "Failed to load sequence"})

        except json.JSONDecodeError as e:
            return json.dumps({"success": False, "error": f"Invalid JSON input: {str(e)}"})
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_parameter_value_recommendations_structured(self, parameter_type: Optional[str] = None) -> str:
        """Get recommended parameter values based on type and available system resources (StructuredTool version)."""
        try:
            # Access the parameter value set generator through RSAP
            if not hasattr(self.rsap, 'action_parameter_value_manager'):
                return json.dumps({
                    "success": False,
                    "error": "Parameter value manager not available in RSAP instance"
                })

            param_manager = self.rsap.action_parameter_value_manager
            
            # Access the parameter values set generator (note: it's "values" not "value")
            if hasattr(param_manager, 'parameter_values_set_generator'):
                value_set_generator = param_manager.parameter_values_set_generator
            elif hasattr(param_manager, 'parameter_value_set_generator'):
                value_set_generator = param_manager.parameter_value_set_generator
            elif hasattr(param_manager, 'value_sets'):
                # Manager is the generator itself
                value_set_generator = param_manager
            else:
                # Debug: log what attributes the manager has
                available_attrs = [attr for attr in dir(param_manager) if not attr.startswith('_')]
                return json.dumps({
                    "success": False,
                    "error": f"Parameter value set generator not available. Available attributes: {available_attrs}"
                })
            
            # Update the value sets to get latest data (TF frames, assembly scene, etc.)
            if hasattr(value_set_generator, 'update'):
                value_set_generator.update()

            # Get value sets
            if parameter_type:
                # Get sets compatible with specific type
                value_set_names = value_set_generator.value_sets.get_all_value_set_names(parameter_type)
            else:
                # Get all value sets
                value_set_names = value_set_generator.value_sets.get_all_value_set_names()

            # Filter out unnecessary value sets (too much information for agent)
            excluded_sets = {'tf_frames', 'vision_cameras', 'vision_processes', 'test_set_1', 'test_set_2', 'test_set_3', 'test_set_4'}
            value_set_names = [name for name in value_set_names if name not in excluded_sets]

            # Build detailed response with actual values
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

            return json.dumps({
                "success": True,
                "parameter_type": parameter_type if parameter_type else "all",
                "value_sets": recommendations,
                "count": len(recommendations)
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
    

    def _get_service_parameters(self, input_str) -> str:
        """Get parameter structure for one or more services."""
        try:
            # Handle case where LLM passes a dict or list directly instead of a JSON string
            if isinstance(input_str, dict):
                service_clients = input_str.get("service_clients", input_str.get("__arg1"))
                if service_clients is None:
                    # Maybe the dict itself is unexpected, try values
                    service_clients = list(input_str.values())[0] if input_str else None
            elif isinstance(input_str, list):
                service_clients = input_str
            elif isinstance(input_str, str):
                # Try parsing as JSON first
                try:
                    params = json.loads(input_str)
                    if isinstance(params, dict):
                        service_clients = params.get("service_clients", params.get("__arg1"))
                    elif isinstance(params, list):
                        service_clients = params
                    else:
                        service_clients = input_str
                except (json.JSONDecodeError, AttributeError):
                    # If it fails, treat input_str as the service client directly
                    service_clients = input_str
            else:
                service_clients = str(input_str)

            if not service_clients:
                return json.dumps({"success": False, "error": "service_clients is required"})

            # Convert single string to list for uniform processing
            if isinstance(service_clients, str):
                service_clients = [service_clients]

            if not isinstance(service_clients, list):
                return json.dumps({"success": False, "error": "service_clients must be a string or list of strings"})

            # Check if services are active (using whitelist filtering like _get_available_services)
            self.rsap.initialize_service_list()
            services = self.rsap.get_active_services()
            filtered_services = self.rsap.get_active_client_whtlist()
            available_clients = [svc[0] for svc in services if svc[0] in filtered_services]
            
            missing_services = [svc for svc in service_clients if svc not in available_clients]
            if missing_services:
                # Check if they exist in unfiltered list
                all_active_clients = [svc[0] for svc in services]
                in_system_but_filtered = [svc for svc in missing_services if svc in all_active_clients]
                
                error_msg = f"Service(s) not found in available services: {', '.join(missing_services)}."
                if in_system_but_filtered:
                    error_msg += f" Note: {', '.join(in_system_but_filtered)} exist but may be filtered by whitelist."
                else:
                    error_msg += " Make sure the service(s) are running."
                
                return json.dumps({
                    "success": False,
                    "error": error_msg,
                    "requested": service_clients,
                    "missing": missing_services,
                    "available_count": len(available_clients)
                })

            # Get parameter structure for requested services
            service_params = self.rsap.get_all_service_req_res_dict(service_clients)

            if not service_params:
                return json.dumps({
                    "success": False,
                    "error": "Could not retrieve parameter information. The service type may not be available or there was an error parsing the service definition.",
                    "requested": service_clients
                })

            return json.dumps({
                "success": True,
                "services": service_params,
                "count": len(service_params)
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_action_at_index(self, input_str: str) -> str:
        """Get details about one or more actions - much more token-efficient than getting full list.
        Note: Accepts 1-based indices (GUI) and converts to 0-based (internal)."""
        try:
            user_indices = None
            
            # Handle multiple input formats from LangChain
            if isinstance(input_str, dict):
                # Check for 'indices' (plural) first, then 'index'
                user_indices = input_str.get("indices") or input_str.get("index")
            elif isinstance(input_str, int):
                # Direct integer - single index
                user_indices = input_str
            elif isinstance(input_str, list):
                # Direct list of integers
                user_indices = input_str
            elif isinstance(input_str, str):
                # Try parsing as JSON first
                try:
                    params = json.loads(input_str)
                    user_indices = params.get("indices") or params.get("index")
                except (json.JSONDecodeError, AttributeError):
                    # Try parsing as direct integer string
                    try:
                        user_indices = int(input_str)
                    except ValueError:
                        pass

            if user_indices is None:
                return json.dumps({"success": False, "error": "index or indices is required"})

            # Normalize to list for uniform processing
            if isinstance(user_indices, int):
                user_indices = [user_indices]
            elif not isinstance(user_indices, list):
                return json.dumps({"success": False, "error": "index must be an integer or list of integers"})

            # Process each index
            results = []
            errors = []
            
            for user_index in user_indices:
                try:
                    user_index = int(user_index)
                    internal_index = user_index - 1  # Convert from 1-based (GUI) to 0-based (internal)

                    if internal_index >= len(self.rsap.action_list) or internal_index < 0:
                        errors.append(f"Index {user_index} out of range (valid: 1-{len(self.rsap.action_list)})")
                        continue

                    action = self.rsap.get_action_at_index(internal_index)
                    
                    action_info = {
                        "index": user_index,  # Return GUI index
                        "name": action.get_name() if hasattr(action, 'get_name') else str(action),
                        "type": type(action).__name__,
                        "is_active": action.is_active() if hasattr(action, 'is_active') else True
                    }
                    
                    # Add client info if available
                    if hasattr(action, 'client'):
                        action_info["client"] = action.client
                    
                    results.append(action_info)
                    
                except (ValueError, TypeError) as e:
                    errors.append(f"Invalid index {user_index}: {str(e)}")

            # Return results
            if len(results) == 1 and len(errors) == 0:
                # Single successful result - return as single object for backward compatibility
                return json.dumps({"success": True, **results[0]})
            else:
                # Multiple results or any errors - return as array
                response = {"success": len(results) > 0, "actions": results}
                if errors:
                    response["errors"] = errors
                return json.dumps(response)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_sequence_summary_structured(self) -> str:
        """Get a lightweight summary of the sequence - just names and indices (StructuredTool version).
        Note: Returns 1-based indices matching GUI display."""
        try:
            summary = {
                "total_count": len(self.rsap.action_list),
                "current_index": self.rsap.get_current_action_index() + 1,  # Convert to 1-based
                "actions": []
            }
            
            for idx, action in enumerate(self.rsap.action_list):
                summary["actions"].append({
                    "index": idx + 1,  # 1-based for GUI
                    "name": action.get_name() if hasattr(action, 'get_name') else str(action),
                    "active": action.is_active() if hasattr(action, 'is_active') else True
                })

            return json.dumps(summary)

        except Exception as e:
            return json.dumps({"error": str(e)})

    def _get_sequence_summary(self, input_str: str = "") -> str:
        """Legacy wrapper for backward compatibility."""
        return self._get_sequence_summary_structured()

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
            return json.dumps({"success": False, "error": str(e)})

    def _load_and_modify_sequence(self, file_path: str) -> str:
        """Load an RSAP sequence from a .rsap.json file."""
        try:
            if not file_path:
                return json.dumps({"success": False, "error": "file_path is required"})

            self.rsap.rsap_file_manager.load_from_JSON(file_path)
            count = len(self.rsap.action_list)
            return json.dumps({
                "success": True,
                "message": f"Loaded sequence with {count} actions from '{file_path}'",
                "action_count": count,
            })
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


