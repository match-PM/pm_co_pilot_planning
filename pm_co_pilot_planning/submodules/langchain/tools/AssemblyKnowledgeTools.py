"""
AssemblyKnowledgeTools: LangChain tools that give the agent knowledge about
the assembly database (components, assemblies, existing RSAP sequences).

These tools allow the agent to:
  - Discover available components and understand their frame structure
    (vision points, laser measurement frames, glue points, gripping point)
  - Discover available assemblies and their component relationships
  - Discover existing RSAP sequences that can be loaded as starting points

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


class ResolveServiceCallInput(BaseModel):
    service_key: str = Field(
        description="The service key from the registry, e.g. 'spawn_component', 'vision_correct_frame'"
    )
    parameter_overrides: str = Field(
        default="{}",
        description='JSON string of parameter overrides, e.g. \'{"frame_name": "UFC_Paper_Vision_Point_1"}\''
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

    def __init__(self, service_node: Node):
        self.service_node = service_node

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

        self.list_available_rsap_sequences_tool = StructuredTool.from_function(
            func=self._list_available_rsap_sequences,
            name="list_available_rsap_sequences",
            description=(
                "Scan the RSAP processes directory and return all existing .rsap.json sequence files. "
                "Use this to discover sequences that can be loaded as starting points "
                "or as reference implementations."
            ),
            args_schema=EmptyInput,
        )

        self.get_service_catalog_tool = StructuredTool.from_function(
            func=self._get_available_services,
            name="get_service_catalog",
            description=(
                "Get the static service catalog: descriptions, parameters, and usage guidance "
                "for all ROS2 assembly services. Use this to understand what services exist "
                "before building a sequence. "
                "Optionally filter by category: motion, scene_management, alignment, "
                "sensing, dispensing, manipulation, curing."
            ),
            args_schema=GetAvailableServicesInput,
        )

        self.resolve_service_call_tool = StructuredTool.from_function(
            func=self._resolve_service_call,
            name="resolve_service_call",
            description=(
                "Resolve a service registry key into a complete service call specification "
                "with merged default + override parameters. Use this when building a sequence "
                "to get the correct client, type, and parameters for an action."
            ),
            args_schema=ResolveServiceCallInput,
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

    def _list_available_rsap_sequences(self) -> str:
        try:
            if not self._rsap_root or not os.path.isdir(self._rsap_root):
                return json.dumps({
                    "success": False,
                    "error": f"RSAP processes directory not found: '{self._rsap_root}'",
                })

            results = []
            for dirpath, _dirnames, filenames in os.walk(self._rsap_root):
                for filename in filenames:
                    if not filename.endswith(".rsap.json"):
                        continue
                    full_path = os.path.join(dirpath, filename)
                    try:
                        with open(full_path, "r") as f:
                            data = json.load(f)
                        results.append({
                            "name": data.get("name", filename.replace(".rsap.json", "")),
                            "file_path": full_path,
                            "action_count": len(data.get("action_list", [])),
                            "saved_at": data.get("saved_at", ""),
                        })
                    except Exception:
                        results.append({
                            "name": filename.replace(".rsap.json", ""),
                            "file_path": full_path,
                            "action_count": None,
                            "saved_at": "",
                        })

            return json.dumps({
                "success": True,
                "count": len(results),
                "sequences": sorted(results, key=lambda x: x["name"]),
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_available_services(self, category: str = "") -> str:
        """Get available services, optionally filtered by category."""
        try:
            from pm_co_pilot_planning.submodules.langchain.tools.ServiceRegistry import ServiceRegistry
            registry = ServiceRegistry()
            if category:
                services = registry.get_services_by_category(category)
                if not services:
                    categories = registry.get_categories()
                    return f"No services found for category '{category}'. Available categories: {categories}"
                # Format subset
                lines = [f"Services in category '{category}':"]
                for key, svc in services.items():
                    lines.append(f"\n  {key}:")
                    lines.append(f"    client: {svc['client']}")
                    lines.append(f"    description: {svc.get('description', '').strip()}")
                    params = svc.get('parameters', {})
                    required = [p for p, info in params.items() if info.get('required')]
                    if required:
                        lines.append(f"    required_params: {required}")
                return "\n".join(lines)
            else:
                return registry.get_service_summary()
        except Exception as e:
            return f"Error getting available services: {e}"

    def _resolve_service_call(self, service_key: str, parameter_overrides: str = "{}") -> str:
        """Resolve a service key with parameter overrides into a ready-to-use call spec."""
        try:
            import json
            overrides = json.loads(parameter_overrides) if parameter_overrides else {}
            from pm_co_pilot_planning.submodules.langchain.tools.ServiceRegistry import ServiceRegistry
            registry = ServiceRegistry()
            result = registry.resolve_service_call(service_key, overrides)
            if result is None:
                all_keys = list(registry.get_all_services().keys())
                return f"Service '{service_key}' not found. Available: {all_keys}"
            return json.dumps(result, indent=2)
        except json.JSONDecodeError as e:
            return f"Invalid JSON in parameter_overrides: {e}"
        except Exception as e:
            return f"Error resolving service call: {e}"
