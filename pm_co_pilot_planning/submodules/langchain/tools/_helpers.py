"""
Shared helper classes and functions for LangChain tool implementations.

Reduces repetitive code across RsapTools, AssemblyKnowledgeTools, and KnowledgeTools:
  - ToolResponse: standardized JSON response builder
  - parse_tool_index / to_internal_index: input parsing and index conversion
  - PoseHelper / FrameHelper / SceneHelper: ROS message serialization
  - ActionHelper: RSAP action info extraction
  - ValueSetHelper: parameter value set generator access
  - generate_sequential_id: knowledge ID generation
  - load_assembly_config: shared config loader
"""

import json
import os
import yaml
from typing import Optional, Dict, Any, List, Tuple

from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

class ToolResponse:
    """Standardized JSON response builder for tool methods."""

    @staticmethod
    def success(**kwargs) -> str:
        return json.dumps({"success": True, **kwargs})

    @staticmethod
    def error(msg: str, **kwargs) -> str:
        return json.dumps({"success": False, "error": msg, **kwargs})


# ---------------------------------------------------------------------------
# Input parsing and index conversion
# ---------------------------------------------------------------------------

def parse_tool_index(input_str, key: str = "index") -> Tuple[Optional[int], Optional[str]]:
    """Parse a user index from various LangChain input formats.

    Handles: dict, int, list, str (JSON or bare integer).
    Returns (user_index, error_msg). error_msg is None on success.
    """
    user_index = None

    if isinstance(input_str, dict):
        user_index = input_str.get(key)
    elif isinstance(input_str, int):
        user_index = input_str
    elif isinstance(input_str, str):
        try:
            params = json.loads(input_str)
            if isinstance(params, dict):
                user_index = params.get(key)
            elif isinstance(params, (int, float)):
                user_index = int(params)
        except (json.JSONDecodeError, AttributeError):
            try:
                user_index = int(input_str)
            except ValueError:
                pass

    if user_index is None:
        return None, f"{key} is required and must be a number"

    return int(user_index), None


def parse_tool_indices(input_str) -> Tuple[Optional[List[int]], Optional[str]]:
    """Parse one or more user indices from various LangChain input formats.

    Handles: dict with 'indices' or 'index' (single or list), int, list, str.
    Returns (user_indices_list, error_msg). error_msg is None on success.
    """
    user_indices = None

    if isinstance(input_str, dict):
        user_indices = input_str.get("indices") or input_str.get("index")
    elif isinstance(input_str, int):
        user_indices = input_str
    elif isinstance(input_str, list):
        user_indices = input_str
    elif isinstance(input_str, str):
        try:
            params = json.loads(input_str)
            if isinstance(params, dict):
                user_indices = params.get("indices") or params.get("index")
            elif isinstance(params, list):
                user_indices = params
            elif isinstance(params, (int, float)):
                user_indices = int(params)
        except (json.JSONDecodeError, AttributeError):
            try:
                user_indices = int(input_str)
            except ValueError:
                pass

    if user_indices is None:
        return None, "index or indices is required"

    # Normalize to list
    if isinstance(user_indices, int):
        user_indices = [user_indices]
    elif not isinstance(user_indices, list):
        return None, "index must be an integer or list of integers"

    return user_indices, None


def to_internal_index(user_index: int, action_count: int) -> Tuple[int, Optional[str]]:
    """Convert 1-based GUI index to 0-based internal index with bounds checking.

    Returns (internal_index, error_msg). error_msg is None if valid.
    """
    internal_index = user_index - 1
    if internal_index < 0 or internal_index >= action_count:
        return -1, f"Index {user_index} out of range (valid: 1-{action_count})"
    return internal_index, None


# ---------------------------------------------------------------------------
# ROS message serialization helpers
# ---------------------------------------------------------------------------

class PoseHelper:
    """Convert ROS Pose messages to dictionaries."""

    @staticmethod
    def to_dict(pose) -> dict:
        return {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
            },
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w,
            },
        }


class FrameHelper:
    """Convert ROS frame messages and properties to dictionaries."""

    @staticmethod
    def relevant_properties(properties) -> dict:
        """Return only the properties relevant to this frame's type."""
        result = {}
        vp = properties.vision_frame_properties
        gp = properties.glue_pt_frame_properties
        lp = properties.laser_frame_properties
        grf = properties.gripping_frame_properties
        af = properties.assembly_frame_properties

        if vp.is_vision_frame:
            result["vision_frame_properties"] = {
                "is_vision_frame": True,
                "has_been_measured": vp.has_been_measured,
            }
        if lp.is_laser_frame:
            result["laser_frame_properties"] = {
                "is_laser_frame": True,
                "has_been_measured": lp.has_been_measured,
            }
        if gp.is_glue_point:
            result["glue_pt_frame_properties"] = {
                "is_glue_point": True,
                "has_been_placed": gp.has_been_placed,
                "has_been_cured": gp.has_been_cured,
                "time_ms": gp.time_ms,
                "dispense_offset_mm": gp.dispense_offset_mm,
            }
        if grf.is_gripping_frame:
            result["gripping_frame_properties"] = {
                "is_gripping_frame": True,
                "compatible_grippers": list(grf.compatible_grippers),
                "compatible_gripper_tips": list(grf.compatible_gripper_tips),
            }
        if af.is_assembly_frame:
            result["assembly_frame_properties"] = {
                "is_assembly_frame": True,
                "is_target_frame": af.is_target_frame,
                "associated_frame": af.associated_frame,
            }
        return result

    @staticmethod
    def to_dict(frame, obj_name=None, detailed=False) -> dict:
        """Serialize a frame to a dict with its relevant type-specific properties."""
        result = {
            "frame_name": frame.frame_name,
            "parent_frame": frame.parent_frame,
            "pose": PoseHelper.to_dict(frame.pose),
            "properties": FrameHelper.relevant_properties(frame.properties),
        }
        if obj_name is not None:
            result["belongs_to_object"] = obj_name
        return result


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------

class SceneHelper:
    """Common scene validation and object lookup patterns."""

    @staticmethod
    def ensure_and_validate(tools_instance) -> Tuple[Optional[Any], Optional[str]]:
        """Ensure scene is updated and return (scene, error_response).

        error_response is a JSON string if scene is unavailable, None otherwise.
        """
        tools_instance._ensure_scene_updated()
        if tools_instance._current_scene is None:
            return None, ToolResponse.error(
                "No scene received yet. The /assembly_manager/scene topic may not be publishing."
            )
        return tools_instance._current_scene, None

    @staticmethod
    def find_object(scene, obj_name: str) -> Tuple[Optional[Any], Optional[str]]:
        """Find an object by name in the scene.

        Returns (obj, error_response). error_response is None if found.
        """
        for obj in scene.objects_in_scene:
            if obj.obj_name == obj_name:
                return obj, None
        available = [o.obj_name for o in scene.objects_in_scene]
        return None, ToolResponse.error(
            f"Object '{obj_name}' not found in scene. Available: {available}"
        )

    @staticmethod
    def find_frame(scene, frame_name: str) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
        """Find a frame by name across all objects in the scene.

        Returns (frame, obj_name_or_None, error_response).
        """
        for obj in scene.objects_in_scene:
            for fr in obj.ref_frames:
                if fr.frame_name == frame_name:
                    return fr, obj.obj_name, None

        # Check scene-level ref_frames
        for fr in scene.ref_frames_in_scene:
            if fr.frame_name == frame_name:
                return fr, None, None

        all_frames = [
            fr.frame_name
            for obj in scene.objects_in_scene
            for fr in obj.ref_frames
        ]
        return None, None, ToolResponse.error(
            f"Frame '{frame_name}' not found in scene.",
            available_frames=all_frames,
        )


# ---------------------------------------------------------------------------
# RSAP action helpers
# ---------------------------------------------------------------------------

class ActionHelper:
    """Extract action info and error details from RSAP actions."""

    @staticmethod
    def to_dict(action, user_index: int) -> dict:
        """Serialize an RSAP action to a dict with 1-based index."""
        info = {
            "index": user_index,
            "name": action.get_name() if hasattr(action, 'get_name') else str(action),
            "type": type(action).__name__,
            "is_active": action.is_active() if hasattr(action, 'is_active') else True,
        }
        if hasattr(action, 'client'):
            info["client"] = action.client
        return info

    @staticmethod
    def extract_srv_response(action) -> dict:
        """Extract error details from a failed action's log entry and response."""
        details = {}
        log_entry = action.get_log_entry() if hasattr(action, 'get_log_entry') else {}
        if log_entry:
            if log_entry.get("service_client"):
                details["service_client"] = log_entry["service_client"]
            if log_entry.get("message"):
                details["message"] = log_entry["message"]
        if hasattr(action, 'response_dict') and action.response_dict:
            response = dict(action.response_dict)
            if "Error" in response:
                details["error_detail"] = response["Error"]
            else:
                details["response"] = response
        return details


# ---------------------------------------------------------------------------
# Value set generator access
# ---------------------------------------------------------------------------

class ValueSetHelper:
    """Access RSAP parameter value set generator with fallback attribute names."""

    @staticmethod
    def get_generator(rsap) -> Tuple[Optional[Any], Optional[str]]:
        """Get the value set generator from RSAP.

        Returns (generator, error_response). error_response is a JSON string on failure.
        """
        if not hasattr(rsap, 'action_parameter_value_manager'):
            return None, ToolResponse.error(
                "Parameter value manager not available in RSAP instance"
            )

        param_manager = rsap.action_parameter_value_manager

        if hasattr(param_manager, 'parameter_values_set_generator'):
            return param_manager.parameter_values_set_generator, None
        elif hasattr(param_manager, 'parameter_value_set_generator'):
            return param_manager.parameter_value_set_generator, None
        elif hasattr(param_manager, 'value_sets'):
            return param_manager, None
        else:
            available_attrs = [attr for attr in dir(param_manager) if not attr.startswith('_')]
            return None, ToolResponse.error(
                f"Parameter value set generator not available. Available attributes: {available_attrs}"
            )


# ---------------------------------------------------------------------------
# Knowledge helpers
# ---------------------------------------------------------------------------

def generate_sequential_id(prefix: str, existing_ids: List[str]) -> str:
    """Generate the next sequential ID like 'prefix_001', 'prefix_002', etc."""
    max_num = 0
    for eid in existing_ids:
        parts = eid.rsplit("_", 1)
        if len(parts) == 2:
            try:
                max_num = max(max_num, int(parts[1]))
            except ValueError:
                pass
    return f"{prefix}_{max_num + 1:03d}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_assembly_config() -> Dict[str, str]:
    """Load assembly_config.yaml from the ROS share directory."""
    package_path = get_package_share_directory("pm_co_pilot_planning")
    config_path = os.path.join(package_path, "assembly_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
