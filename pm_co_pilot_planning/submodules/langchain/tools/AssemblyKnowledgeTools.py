"""
AssemblyKnowledgeTools: LangChain tools that give the agent knowledge about
the assembly database (components, assemblies).

These tools allow the agent to:
  - Discover available components and understand their frame structure
    (vision points, laser measurement frames, glue points, gripping point)
  - Discover available assemblies and their component relationships

Frame naming convention (enforced by assembly_manager when spawning):
  {ComponentName}_{FrameName}
  e.g.  UFC_Paper  +  Vision_Point_1  →  UFC_Paper_Vision_Point_1
"""

import json
import os
import re
import yaml
from typing import Optional, List, Dict, Any

from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
import assembly_manager_interfaces.msg as am_msgs
import rclpy
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from ros_sequential_action_programmer.submodules.RosSequentialActionProgrammer import RosSequentialActionProgrammer


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class GetComponentDescriptionInput(BaseModel):
    file_path: str = Field(
        description="Absolute path to a component .json file, "
                    "or just the component name (e.g. 'UFC_Paper') to search the database."
    )


class GetAssemblyDescriptionInput(BaseModel):
    file_path: str = Field(
        description="Absolute path to an assembly .json file, "
                    "or just the assembly name to search the database."
    )


class EmptyInput(BaseModel):
    pass


class GetAvailableServicesInput(BaseModel):
    category: str = Field(
        default="",
        description="Optional category filter: motion, scene_management, alignment, sensing, dispensing, manipulation, curing. Empty string returns all."
    )


class GetObjectPropertiesInput(BaseModel):
    obj_name: str = Field(
        description="Name of the object in the scene, e.g. 'UFC_Paper'."
    )


class GetObjectFramesInput(BaseModel):
    obj_name: str = Field(
        description="Name of the object in the scene, e.g. 'UFC_Paper'."
    )


class GetFramePropertiesInput(BaseModel):
    frame_name: str = Field(
        description="Full frame name as it appears in the scene, e.g. 'UFC_Paper_Vision_Point_1'."
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_assembly_config() -> Dict[str, str]:
    """Load assembly_config.yaml from the ROS share directory."""
    package_path = get_package_share_directory("pm_co_pilot_planning")
    config_path = os.path.join(package_path, "assembly_config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _categorize_frames(ref_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Categorise ref_frames from a component mountingDescription into groups
    based on naming conventions used in the PM assembly system.

    Returns a dict with keys:
        vision_points, laser_measurement_frames, glue_points,
        gripping_point, other_frames
    """
    vision_points = []
    laser_frames = []
    glue_points = []
    gripping_point = None
    other_frames = []

    for frame in ref_frames:
        name = frame.get("name", "")
        # Skip helper frames (used only for plane/axis definitions)
        if "_helper" in name.lower():
            continue

        if re.search(r"Vision_Point", name, re.IGNORECASE):
            vision_points.append(name)
        elif re.search(r"Laser_Mes_Frame|Laser_Frame|Laser_Mes", name, re.IGNORECASE):
            laser_frames.append(name)
        elif re.search(r"Glue_Point|Glue_Frame|_Glue_", name, re.IGNORECASE):
            glue_points.append(name)
        elif re.search(r"Gripping_Point|Grip_Point", name, re.IGNORECASE):
            gripping_point = name
        else:
            other_frames.append(name)

    return {
        "vision_points": sorted(vision_points),
        "laser_measurement_frames": sorted(laser_frames),
        "glue_points": sorted(glue_points),
        "gripping_point": gripping_point,
        "other_frames": sorted(other_frames),
    }


def _determine_gonio_side(spawning_origin: str) -> str:
    """Infer gonio side from the spawningOrigin value."""
    if not spawning_origin:
        return "unknown"
    origin_lower = spawning_origin.lower()
    if "right" in origin_lower:
        return "right"
    if "left" in origin_lower:
        return "left"
    return "unknown"


def _scan_json_files(root_dir: str, expected_type: str) -> List[Dict[str, str]]:
    """
    Walk root_dir recursively, return list of {name, file_path} for every
    .json file whose top-level "type" field matches expected_type.
    Skips files that cannot be parsed.
    """
    results = []
    if not os.path.isdir(root_dir):
        return results

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            full_path = os.path.join(dirpath, filename)
            try:
                with open(full_path, "r") as f:
                    data = json.load(f)
                if data.get("type") == expected_type:
                    results.append({
                        "name": data.get("name", filename.replace(".json", "")),
                        "file_path": full_path,
                    })
            except Exception:
                continue
    return results


def _resolve_file_path(query: str, expected_type: str, search_root: str) -> Optional[str]:
    """
    If `query` looks like an absolute path, return it directly.
    Otherwise search the database for a file whose name matches `query`.
    """
    if os.path.isabs(query) and os.path.isfile(query):
        return query

    # Search by name
    candidates = _scan_json_files(search_root, expected_type)
    query_lower = query.lower().replace(" ", "_")
    for c in candidates:
        if c["name"].lower().replace(" ", "_") == query_lower:
            return c["file_path"]
        if os.path.basename(c["file_path"]).replace(".json", "").lower() == query_lower:
            return c["file_path"]
    return None


# ---------------------------------------------------------------------------
# AssemblyKnowledgeTools class
# ---------------------------------------------------------------------------

class AssemblyKnowledgeTools:
    """
    Provides LangChain tools for querying the PM assembly database.
    Pass a ROS node for logging; RSAP instance is not required.
    """

    def __init__(self, service_node: Node, rsap_instance = None):
        self.service_node = service_node
        self.service_node.get_logger().info("Initializing AssemblyKnowledgeTools...")
        self._current_scene: Optional[am_msgs.ObjectScene] = None
        if rsap_instance:
            self.rsap = rsap_instance
        else:
            self.rsap = RosSequentialActionProgrammer(service_node)

        # Subscribe to the live assembly scene
        self._scene_sub = self.service_node.create_subscription(
            am_msgs.ObjectScene,
            "/assembly_manager/scene",
            self._scene_callback,
            10,
        )
        self.service_node.get_logger().info("Subscribed to /assembly_manager/scene for live scene updates.")

        try:
            cfg = _load_assembly_config()
            self._db_root = cfg.get("assembly_database_path", "")
            self._components_root = os.path.join(
                self._db_root, cfg.get("components_subdir", "Assembly_Part_Data")
            )
            self._rsap_root = os.path.join(
                self._db_root, cfg.get("rsap_processes_subdir", "RSAP_Processes")
            )
        except Exception as e:
            self.service_node.get_logger().warning(
                f"AssemblyKnowledgeTools: could not load assembly_config.yaml: {e}"
            )
            self._db_root = ""
            self._components_root = ""
            self._rsap_root = ""

        # ---- Tool definitions ----

        self.list_available_components_tool = StructuredTool.from_function(
            func=self._list_available_components,
            name="list_available_components",
            description=(
                "Scan the assembly database and return all available component files. "
                "Each entry contains the component name, file path, and which gonio stage it belongs to "
                "(left/right, derived from spawningOrigin). "
                "Use this to discover what components exist before planning an assembly sequence."
            ),
            args_schema=EmptyInput,
        )

        self.get_component_description_tool = StructuredTool.from_function(
            func=self._get_component_description,
            name="get_component_description",
            description=(
                "Read a component description JSON file and return a structured summary of its frames, "
                "categorised by purpose:\n"
                "  - vision_points: frames used for vision-based correction\n"
                "  - laser_measurement_frames: frames used for gonio alignment\n"
                "  - glue_points: frames where adhesive should be dispensed\n"
                "  - gripping_point: the frame used to grip the component\n"
                "  - spawning_origin: determines gonio side (left/right)\n\n"
                "Input: absolute file path OR just the component name (e.g. 'UFC_Paper').\n"
                "CRITICAL: When a component is spawned, all its frame names are prefixed with "
                "the component name: {{ComponentName}}_{{FrameName}}. "
                "Example: 'UFC_Paper' + 'Vision_Point_1' → 'UFC_Paper_Vision_Point_1'."
            ),
            args_schema=GetComponentDescriptionInput,
        )

        self.list_available_assemblies_tool = StructuredTool.from_function(
            func=self._list_available_assemblies,
            name="list_available_assemblies",
            description=(
                "Scan the assembly database and return all available assembly description files. "
                "Each entry contains the assembly name, file path, and list of component names it contains. "
                "Use this to find assembly description files needed for 'Create Assembly Instruction' actions."
            ),
            args_schema=EmptyInput,
        )

        self.get_assembly_description_tool = StructuredTool.from_function(
            func=self._get_assembly_description,
            name="get_assembly_description",
            description=(
                "Read an assembly description JSON file and return its component list "
                "and a summary of assembly constraints. "
                "Input: absolute file path OR just the assembly name (e.g. 'Baugruppe_S17')."
            ),
            args_schema=GetAssemblyDescriptionInput,
        )

        self.list_objects_in_scene_tool = StructuredTool.from_function(
            func=self._list_objects_in_scene,
            name="list_objects_in_scene",
            description=(
                "Return all objects currently present in the live assembly scene. "
                "Each entry contains the object name, its parent frame, and properties. "
                "Use this to see what has already been spawned before planning actions."
            ),
            args_schema=EmptyInput,
        )

        self.get_object_properties_tool = StructuredTool.from_function(
            func=self._get_object_properties,
            name="get_object_properties",
            description=(
                "Return the properties of a specific object in the live scene. "
                "Input: the object name as it appears in the scene (e.g. 'UFC_Paper')."
            ),
            args_schema=GetObjectPropertiesInput,
        )

        self.get_object_frames_tool = StructuredTool.from_function(
            func=self._get_object_frames,
            name="get_object_frames",
            description=(
                "Return all reference frames that belong to a specific object in the live scene. "
                "Each frame entry includes the frame name, parent frame, pose (position + orientation), "
                "and a summary of its properties (vision, laser, glue, gripping, assembly). "
                "Input: the object name as it appears in the scene (e.g. 'UFC_Paper')."
            ),
            args_schema=GetObjectFramesInput,
        )

        self.get_frame_properties_tool = StructuredTool.from_function(
            func=self._get_frame_properties,
            name="get_frame_properties",
            description=(
                "Return the full properties of a specific frame in the live scene, identified by its "
                "exact frame name (e.g. 'UFC_Paper_Vision_Point_1'). "
                "Properties include vision, laser, glue, gripping, and assembly frame flags and values. "
                "Searches across all objects in the scene."
            ),
            args_schema=GetFramePropertiesInput,
        )

        self.get_frames_in_scene_tool = StructuredTool.from_function(
            func=self._get_frames_in_scene,
            name="get_frames_in_scene", 
            description=(
                "Return all frames currently present in the live assembly scene. "
                "frames_in_scene and tf_frames"
            ),
            args_schema=EmptyInput,
        )

    # ------------------------------------------------------------------
    # Scene subscriber callback
    # ------------------------------------------------------------------

    def _scene_callback(self, msg: am_msgs.ObjectScene) -> None:
        """Store the latest scene message received from /assembly_manager/scene."""
        self._current_scene = msg

    def _ensure_scene_updated(self, timeout_sec: float = 0.5) -> None:
        """
        Spin the node once for a short duration to process pending subscription messages.
        This ensures the scene callback has a chance to be called if a message is available.
        """
        try:
            rclpy.spin_once(self.service_node, timeout_sec=timeout_sec)
        except Exception as e:
            self.service_node.get_logger().warning(
                f"AssemblyKnowledgeTools: could not spin node: {e}"
            )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    def _list_available_components(self) -> str:
        try:
            if not self._components_root:
                return json.dumps({"error": "Assembly database path not configured"})

            candidates = _scan_json_files(self._components_root, "Component")
            results = []
            for c in candidates:
                gonio_side = "unknown"
                try:
                    with open(c["file_path"], "r") as f:
                        data = json.load(f)
                    mounting = data.get("mountingDescription", {})
                    refs = mounting.get("mountingReferences", {})
                    spawning_origin = refs.get("spawningOrigin", "")
                    gonio_side = _determine_gonio_side(spawning_origin)
                except Exception:
                    pass

                results.append({
                    "name": c["name"],
                    "file_path": c["file_path"],
                    "gonio_side": gonio_side,
                })

            return json.dumps({
                "success": True,
                "count": len(results),
                "components": sorted(results, key=lambda x: x["name"]),
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_component_description(self, file_path: str) -> str:
        try:
            resolved = _resolve_file_path(file_path, "Component", self._components_root)
            if resolved is None:
                return json.dumps({
                    "success": False,
                    "error": f"Component not found: '{file_path}'. "
                             "Use list_available_components to see what is available.",
                })

            with open(resolved, "r") as f:
                data = json.load(f)

            mounting = data.get("mountingDescription", {})
            refs = mounting.get("mountingReferences", {})
            spawning_origin = refs.get("spawningOrigin", "")
            gonio_side = _determine_gonio_side(spawning_origin)

            ref_frames = refs.get("ref_frames", [])
            frame_categories = _categorize_frames(ref_frames)

            component_name = data.get("name", os.path.basename(resolved).replace(".json", ""))

            # Build prefixed frame names (as they will appear after spawning)
            def prefix(frames):
                if isinstance(frames, list):
                    return [f"{component_name}_{f}" for f in frames]
                if frames:
                    return f"{component_name}_{frames}"
                return None

            summary = {
                "success": True,
                "name": component_name,
                "file_path": resolved,
                "spawning_origin": spawning_origin,
                "gonio_side": gonio_side,
                "gonio_service": (
                    "/pm_skills/iterative_align_gonio_right"
                    if gonio_side == "right"
                    else "/pm_skills/iterative_align_gonio_left"
                    if gonio_side == "left"
                    else "unknown"
                ),
                "frames": {
                    "vision_points": frame_categories["vision_points"],
                    "laser_measurement_frames": frame_categories["laser_measurement_frames"],
                    "glue_points": frame_categories["glue_points"],
                    "gripping_point": frame_categories["gripping_point"],
                    "other_frames": frame_categories["other_frames"],
                },
                "spawned_frame_names": {
                    "vision_points": prefix(frame_categories["vision_points"]),
                    "laser_measurement_frames": prefix(frame_categories["laser_measurement_frames"]),
                    "glue_points": prefix(frame_categories["glue_points"]),
                    "gripping_point": prefix(frame_categories["gripping_point"]),
                },
                "roles": {
                    "is_base_component": len(frame_categories["glue_points"]) > 0,
                    "is_placed_component": frame_categories["gripping_point"] is not None,
                    "has_vision_correction": len(frame_categories["vision_points"]) > 0,
                    "has_gonio_alignment": len(frame_categories["laser_measurement_frames"]) > 0,
                },
            }

            return json.dumps(summary)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _list_available_assemblies(self) -> str:
        try:
            if not self._components_root:
                return json.dumps({"error": "Assembly database path not configured"})

            candidates = _scan_json_files(self._components_root, "Assembly")
            results = []
            for c in candidates:
                component_names = []
                try:
                    with open(c["file_path"], "r") as f:
                        data = json.load(f)
                    component_names = [
                        comp.get("name", "") for comp in data.get("components", [])
                    ]
                except Exception:
                    pass

                results.append({
                    "name": c["name"],
                    "file_path": c["file_path"],
                    "components": component_names,
                })

            return json.dumps({
                "success": True,
                "count": len(results),
                "assemblies": sorted(results, key=lambda x: x["name"]),
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_assembly_description(self, file_path: str) -> str:
        try:
            resolved = _resolve_file_path(file_path, "Assembly", self._components_root)
            if resolved is None:
                return json.dumps({
                    "success": False,
                    "error": f"Assembly not found: '{file_path}'. "
                             "Use list_available_assemblies to see what is available.",
                })

            with open(resolved, "r") as f:
                data = json.load(f)

            components = [
                {"name": c.get("name", ""), "guid": c.get("guid", "")}
                for c in data.get("components", [])
            ]

            constraints = []
            for c in data.get("assemblyConstraints", []):
                constraints.append({
                    "name": c.get("name", ""),
                    "plane_matches": len(c.get("planeMatchConstraints", [])),
                })

            assembly_frames = [
                f.get("name", "") for f in data.get("ref_frames", [])
            ]

            return json.dumps({
                "success": True,
                "name": data.get("name", ""),
                "file_path": resolved,
                "components": components,
                "assembly_constraints": constraints,
                "assembly_frames": assembly_frames,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    # ------------------------------------------------------------------
    # Live scene tools
    # ------------------------------------------------------------------

    def _list_objects_in_scene(self) -> str:
        """Return all objects currently in the live assembly scene."""
        try:
            self._ensure_scene_updated()
            if self._current_scene is None:
                return json.dumps({
                    "success": False,
                    "error": "No scene received yet. The /assembly_manager/scene topic may not be publishing.",
                })

            objects = []
            for obj in self._current_scene.objects_in_scene:
                objects.append({
                    "obj_name": obj.obj_name,
                    "parent_frame": obj.parent_frame,
                        "properties": {
                            "is_gripped": obj.properties.is_gripped,
                            "is_assembled": obj.properties.is_assembled,
                        },
                })

            return json.dumps({
                "success": True,
                "count": len(objects),
                "objects": objects,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
        
    def _get_object_properties(self, obj_name: str) -> str:
        """Return properties of a named object in the live scene."""
        try:
            self._ensure_scene_updated()
            if self._current_scene is None:
                return json.dumps({
                    "success": False,
                    "error": "No scene received yet. The /assembly_manager/scene topic may not be publishing.",
                })

            for obj in self._current_scene.objects_in_scene:
                if obj.obj_name == obj_name:
                    return json.dumps({
                        "success": True,
                        "obj_name": obj.obj_name,
                        "parent_frame": obj.parent_frame,
                        "properties": {
                            "is_gripped": obj.properties.is_gripped,
                            "is_assembled": obj.properties.is_assembled,
                        },
                    })

            available = [o.obj_name for o in self._current_scene.objects_in_scene]
            return json.dumps({
                "success": False,
                "error": f"Object '{obj_name}' not found in scene. Available: {available}",
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_object_frames(self, obj_name: str) -> str:
        """Return all reference frames for a named object in the live scene."""
        try:
            self._ensure_scene_updated()
            if self._current_scene is None:
                return json.dumps({
                    "success": False,
                    "error": "No scene received yet. The /assembly_manager/scene topic may not be publishing.",
                })

            target_obj = None
            for obj in self._current_scene.objects_in_scene:
                if obj.obj_name == obj_name:
                    target_obj = obj
                    break

            if target_obj is None:
                available = [o.obj_name for o in self._current_scene.objects_in_scene]
                return json.dumps({
                    "success": False,
                    "error": f"Object '{obj_name}' not found in scene. Available: {available}",
                })

            frames = []
            for fr in target_obj.ref_frames:
                p = fr.properties
                frames.append({
                    "frame_name": fr.frame_name,
                    "parent_frame": fr.parent_frame,
                    "pose": {
                        "position": {
                            "x": fr.pose.position.x,
                            "y": fr.pose.position.y,
                            "z": fr.pose.position.z,
                        },
                        "orientation": {
                            "x": fr.pose.orientation.x,
                            "y": fr.pose.orientation.y,
                            "z": fr.pose.orientation.z,
                            "w": fr.pose.orientation.w,
                        },
                    },
                    "properties_summary": {
                        "is_vision_frame": p.vision_frame_properties.is_vision_frame,
                        "is_laser_frame": p.laser_frame_properties.is_laser_frame,
                        "is_glue_point": p.glue_pt_frame_properties.is_glue_point,
                        "is_gripping_frame": p.gripping_frame_properties.is_gripping_frame,
                        "is_assembly_frame": p.assembly_frame_properties.is_assembly_frame,
                        "is_target_frame": p.assembly_frame_properties.is_target_frame,
                    },
                })

            return json.dumps({
                "success": True,
                "obj_name": obj_name,
                "frame_count": len(frames),
                "frames": frames,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_frame_properties(self, frame_name: str) -> str:
        """Return full properties of a named frame from any object in the live scene."""
        try:
            self._ensure_scene_updated()
            if self._current_scene is None:
                return json.dumps({
                    "success": False,
                    "error": "No scene received yet. The /assembly_manager/scene topic may not be publishing.",
                })

            for obj in self._current_scene.objects_in_scene:
                for fr in obj.ref_frames:
                    if fr.frame_name == frame_name:
                        p = fr.properties
                        vp = p.vision_frame_properties
                        gp = p.glue_pt_frame_properties
                        lp = p.laser_frame_properties
                        grf = p.gripping_frame_properties
                        af = p.assembly_frame_properties
                        return json.dumps({
                            "success": True,
                            "frame_name": fr.frame_name,
                            "parent_frame": fr.parent_frame,
                            "belongs_to_object": obj.obj_name,
                            "pose": {
                                "position": {
                                    "x": fr.pose.position.x,
                                    "y": fr.pose.position.y,
                                    "z": fr.pose.position.z,
                                },
                                "orientation": {
                                    "x": fr.pose.orientation.x,
                                    "y": fr.pose.orientation.y,
                                    "z": fr.pose.orientation.z,
                                    "w": fr.pose.orientation.w,
                                },
                            },
                            "properties": {
                                "vision_frame": {
                                    "is_vision_frame": vp.is_vision_frame,
                                    "has_been_measured": vp.has_been_measured,
                                },
                                "glue_pt_frame": {
                                    "is_glue_point": gp.is_glue_point,
                                    "has_been_placed": gp.has_been_placed,
                                    "has_been_cured": gp.has_been_cured,
                                    "time_ms": gp.time_ms,
                                    "dispense_offset_mm": gp.dispense_offset_mm,
                                },
                                "laser_frame": {
                                    "is_laser_frame": lp.is_laser_frame,
                                    "has_been_measured": lp.has_been_measured,
                                },
                                "gripping_frame": {
                                    "is_gripping_frame": grf.is_gripping_frame,
                                    "compatible_grippers": list(grf.compatible_grippers),
                                    "compatible_gripper_tips": list(grf.compatible_gripper_tips),
                                },
                                "assembly_frame": {
                                    "is_assembly_frame": af.is_assembly_frame,
                                    "is_target_frame": af.is_target_frame,
                                    "associated_frame": af.associated_frame,
                                },
                            },
                        })

            # Also check scene-level ref_frames (not attached to any object)
            for fr in self._current_scene.ref_frames_in_scene:
                if fr.frame_name == frame_name:
                    return json.dumps({
                        "success": True,
                        "frame_name": fr.frame_name,
                        "parent_frame": fr.parent_frame,
                        "belongs_to_object": None,
                        "note": "This is a scene-level frame (not attached to a specific object).",
                        "pose": {
                            "position": {
                                "x": fr.pose.position.x,
                                "y": fr.pose.position.y,
                                "z": fr.pose.position.z,
                            },
                            "orientation": {
                                "x": fr.pose.orientation.x,
                                "y": fr.pose.orientation.y,
                                "z": fr.pose.orientation.z,
                                "w": fr.pose.orientation.w,
                            },
                        },
                    })

            all_frames = [
                fr.frame_name
                for obj in self._current_scene.objects_in_scene
                for fr in obj.ref_frames
            ]
            return json.dumps({
                "success": False,
                "error": f"Frame '{frame_name}' not found in scene.",
                "available_frames": all_frames,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})


    def _get_frames_in_scene(self, parameter_type: Optional[str] = None) -> str:
        """Get frames in the current scene. frames_in_scene + tf_frames"""
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
            excluded_sets = {'components_in_scene','instructions_in_scene','vision_cameras', 'vision_processes', 'test_set_1', 'test_set_2', 'test_set_3', 'test_set_4'}
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