"""
KnowledgeTools: LangChain tools for service-centric domain knowledge management.

Provides five tools:
  - query_assembly_knowledge: Retrieve service knowledge (parameters, usage notes)
  - get_similar_assembly_example: Find reference execution traces by component/assembly name
  - record_knowledge: Save new knowledge to a specific service or general knowledge
  - confirm_knowledge: Increment confirmation_count on an existing usage_note by id
  - contradict_knowledge: Increment contradiction_count on an existing usage_note by id
"""

import json
import os
import re
import threading
import yaml
from datetime import date, datetime
from typing import Optional, List, Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from rclpy.node import Node

from pm_co_pilot_planning.submodules.langchain.tools._helpers import (
    ToolResponse, generate_sequential_id, load_assembly_config,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryKnowledgeInput(BaseModel):
    service_name: str = Field(
        default="",
        description=(
            "Optional: exact ROS2 service_client to query knowledge for "
            "(e.g. '/pm_skills/vision_correct_frame'). "
            "If empty, returns ALL service knowledge and general knowledge."
        ),
    )


class GetExampleInput(BaseModel):
    component_names: List[str] = Field(
        default_factory=list,
        description="List of component names to search for in example traces.",
    )
    assembly_name: str = Field(
        default="",
        description="Assembly name to search for. If provided, matches examples for this assembly.",
    )


class ConfirmKnowledgeInput(BaseModel):
    service_name: str = Field(
        description=(
            "Exact ROS2 service_client whose usage_note is being confirmed "
            "(e.g. '/pm_skills/vision_correct_frame')."
        ),
    )
    note_id: str = Field(
        description=(
            "The id of the existing usage_note to confirm "
            "(e.g. 'usage_vision_correct_frame_001')."
        ),
    )


class ContradictKnowledgeInput(BaseModel):
    service_name: str = Field(
        description=(
            "Exact ROS2 service_client whose usage_note is being contradicted "
            "(e.g. '/pm_skills/iterative_align_gonio_left')."
        ),
    )
    note_id: str = Field(
        description="The id of the existing usage_note that the new evidence disproves.",
    )
    evidence: str = Field(
        description=(
            "One-sentence description of the specific state change, parameter value, "
            "or error message that contradicts this note."
        ),
    )


class RecordKnowledgeInput(BaseModel):
    service_name: str = Field(
        default="",
        description=(
            "Exact ROS2 service_client to record knowledge for "
            "(e.g. '/pm_skills/iterative_gonio_align'). "
            "If empty, records to general_knowledge."
        ),
    )
    field: str = Field(
        description=(
            "Which field to add to: 'usage_notes' or 'parameters'. "
            "For general_knowledge (empty service_name), use 'usage_notes'. "
            "For 'parameters': content must be a JSON string with keys 'name', 'type', "
            "and 'description' (e.g. '{\"name\": \"target_frame\", \"type\": \"string\", \"description\": \"Frame to align to\"}')."
        ),
    )
    content: str = Field(
        description=(
            "The knowledge to record. For usage_notes: a clear prescriptive statement "
            "(what a future planner MUST, SHOULD, or MUST NOT do). "
            "For parameters: a JSON string with 'name', 'type', and 'description' keys."
        ),
    )
    category: str = Field(
        default="",
        description="Category for general_knowledge entries (e.g. 'component_roles', 'assembly_patterns').",
    )
    source: str = Field(
        default="experience",
        description=(
            "Source of this knowledge: 'user_correction' (user explicitly corrected you, confidence=0.9) "
            "or 'experience' (learned from execution, confidence=0.6-0.7)."
        ),
    )
    confidence: float = Field(
        default=0.7,
        description="Confidence level 0.0-1.0. User corrections: 0.9, resolved errors: 0.7, discovered patterns: 0.6.",
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_knowledge(knowledge_path: str) -> Dict[str, Any]:
    """Load the service_knowledge.yaml file. Returns empty structure if not found."""
    if not os.path.isfile(knowledge_path):
        return {
            "metadata": {"version": 2, "schema": "service_knowledge_v2"},
            "services": {},
            "general_knowledge": [],
        }
    with open(knowledge_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {
            "metadata": {"version": 2, "schema": "service_knowledge_v2"},
            "services": {},
            "general_knowledge": [],
        }
    # YAML serialises an empty dict as `{}` but can round-trip as `[]` when
    # the file was written with default_flow_style or by hand. Normalise so
    # _record_knowledge never receives a list where it expects a dict.
    if not isinstance(data.get("services"), dict):
        data["services"] = {}
    if not isinstance(data.get("general_knowledge"), list):
        data["general_knowledge"] = []
    return data


def _save_knowledge(knowledge_path: str, data: Dict[str, Any]) -> None:
    """Save the knowledge data back to service_knowledge.yaml."""
    data["metadata"]["last_updated"] = date.today().isoformat()
    with open(knowledge_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


_SOURCE_PRIORITY = {"user_correction": 3, "experience": 2, "bootstrap": 1}


def _normalize(text: str) -> str:
    """Lowercase and collapse all non-alphanumeric runs to a single space for dedup."""
    return re.sub(r"[\s\W_]+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# KnowledgeTools class
# ---------------------------------------------------------------------------

class KnowledgeTools:
    """
    Provides LangChain tools for querying and updating the service-centric knowledge base.
    """

    def __init__(self, service_node: Node):
        self.service_node = service_node
        self._write_lock = threading.Lock()
        self.service_node.get_logger().info("Initializing KnowledgeTools...")

        try:
            cfg = load_assembly_config()
            self._db_root = cfg.get("assembly_database_path", "")
            kb_subdir = cfg.get("knowledge_base_subdir", "Knowledge_Base")
            self._kb_root = os.path.join(self._db_root, kb_subdir)
            self._knowledge_path = os.path.join(self._kb_root, "service_knowledge.yaml")
            self._examples_dir = os.path.join(self._kb_root, "examples")
        except Exception as e:
            self.service_node.get_logger().warning(
                f"KnowledgeTools: could not load assembly_config.yaml: {e}"
            )
            self._db_root = ""
            self._kb_root = ""
            self._knowledge_path = ""
            self._examples_dir = ""

        # Ensure directories exist
        os.makedirs(self._examples_dir, exist_ok=True)

        self.service_node.get_logger().info(
            f"KnowledgeTools initialized. Knowledge base: {self._kb_root}"
        )

        # ---- Tool definitions ----

        self.query_assembly_knowledge_tool = StructuredTool.from_function(
            func=self._query_assembly_knowledge,
            name="query_assembly_knowledge",
            description=(
                "Return service domain knowledge: parameters, usage notes, and general knowledge.\n\n"
                "- Call with NO arguments to get ALL services + general knowledge (for planning).\n"
                "- Call with service_name to get knowledge for a specific service (for error recovery).\n\n"
                "ALWAYS call this FIRST when planning a new assembly sequence."
            ),
            args_schema=QueryKnowledgeInput,
        )

        self.get_similar_assembly_example_tool = StructuredTool.from_function(
            func=self._get_similar_assembly_example,
            name="get_similar_assembly_example",
            description=(
                "Search the knowledge base for reference execution traces of similar "
                "assemblies. Returns a step-by-step sequence pattern from a previously "
                "successful assembly session.\n\n"
                "Use this when building a complete assembly sequence to find a proven "
                "pattern to follow. Search by component names or assembly name."
            ),
            args_schema=GetExampleInput,
        )

        self.record_knowledge_tool = StructuredTool.from_function(
            func=self._record_knowledge,
            name="record_knowledge",
            description=(
                "Save NEW knowledge to the service knowledge base.\n\n"
                "Specify a service_client to add to a specific service's entry, "
                "or leave empty to add to general_knowledge.\n\n"
                "Fields you can add to:\n"
                "  - 'usage_notes': add a prescriptive rule, constraint, or learned lesson "
                "(what a future planner MUST, SHOULD, or MUST NOT do)\n"
                "  - 'parameters': add or update one parameter entry; content must be a JSON string: "
                "{\"name\": \"<param_name>\", \"type\": \"<type>\", \"description\": \"<what it does>\"}\n\n"
                "Call this ONLY for genuinely new insights not covered by existing notes.\n"
                "To reinforce an existing note use confirm_knowledge; to mark one disproven "
                "use contradict_knowledge.\n\n"
                "Call this when:\n"
                "  - The user corrects your approach (source='user_correction', confidence=0.9)\n"
                "  - You resolve an execution error (source='experience', confidence=0.7)\n"
                "  - You discover a working pattern (source='experience', confidence=0.6)\n\n"
                "Do NOT record facts discoverable via existing tools (component frames, service parameters)."
            ),
            args_schema=RecordKnowledgeInput,
        )

        self.confirm_knowledge_tool = StructuredTool.from_function(
            func=self._confirm_knowledge,
            name="confirm_knowledge",
            description=(
                "Confirm that an existing usage_note is consistent with the current run's evidence.\n\n"
                "Call this when the execution evidence re-demonstrates a rule that is already "
                "in the knowledge base. Increments confirmation_count and bumps confidence (+0.05).\n\n"
                "Use the note id from EXISTING KB STATE (e.g. 'usage_vision_correct_frame_001').\n"
                "Do NOT call this for general_knowledge entries — they are authoritative by definition."
            ),
            args_schema=ConfirmKnowledgeInput,
        )

        self.contradict_knowledge_tool = StructuredTool.from_function(
            func=self._contradict_knowledge,
            name="contradict_knowledge",
            description=(
                "Mark an existing usage_note as contradicted by the current run's evidence.\n\n"
                "Call this when the execution evidence disproves a rule already in the KB "
                "(e.g. a note says X is required but this run succeeded without X; "
                "or a note specifies a parameter value that this run showed was wrong).\n\n"
                "Increments contradiction_count, decrements confidence (-0.15), and logs the evidence.\n"
                "Use the note id from EXISTING KB STATE. "
                "Do NOT call this for general_knowledge entries."
            ),
            args_schema=ContradictKnowledgeInput,
        )

    # ------------------------------------------------------------------
    # Public helpers (called from Agent, not exposed as LLM tools)
    # ------------------------------------------------------------------

    def get_knowledge_for_services(self, service_names: List[str]) -> str:
        """Return KB entries for only the given services + general_knowledge as JSON.

        Used by the post-execution learning nudge to inject targeted context
        into the prompt instead of having the LLM query the full KB.
        """
        data = _load_knowledge(self._knowledge_path)
        services_kb = data.get("services", {})
        general = data.get("general_knowledge", [])

        subset = {}
        for svc in service_names:
            if svc in services_kb:
                entry = services_kb[svc]
                subset[svc] = {
                    "description": entry.get("description", ""),
                    "usage_notes": [
                        {"id": n.get("id"), "note": n.get("note", "")}
                        for n in entry.get("usage_notes", [])
                        if isinstance(n, dict) and n.get("note")
                    ],
                }
            else:
                subset[svc] = {"not_in_kb": True}

        return json.dumps(
            {"services": subset, "general_knowledge": general},
            indent=2, default=str,
        )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    def _query_assembly_knowledge(self, service_name: str = "") -> str:
        """Return service knowledge, optionally filtered to a specific service."""
        try:
            data = _load_knowledge(self._knowledge_path)
            services = data.get("services", {})
            general = data.get("general_knowledge", [])

            if service_name:
                if service_name not in services:
                    return ToolResponse.success(
                        service=service_name,
                        found=False,
                        note=f"No knowledge entry for '{service_name}'. You may need to reason from first principles or check get_available_services.",
                    )

                entry = services[service_name]
                return ToolResponse.success(
                    service=service_name,
                    found=True,
                    description=entry.get("description", ""),
                    usage_notes=entry.get("usage_notes", []),
                )
            else:
                formatted_services = {}
                for svc_name, entry in services.items():
                    formatted_services[svc_name] = {
                        "description": entry.get("description", ""),
                        "usage_notes": entry.get("usage_notes", []),
                    }

                documented = list(services.keys())
                return ToolResponse.success(
                    knowledge_coverage=f"{len(documented)} services have documented knowledge",
                    documented_services=documented,
                    service_count=len(formatted_services),
                    services=formatted_services,
                    general_knowledge=general,
                )

        except Exception as e:
            return ToolResponse.error(str(e))

    def _get_similar_assembly_example(
        self, component_names: List[str] = None, assembly_name: str = ""
    ) -> str:
        """Search for reference execution traces matching component or assembly names."""
        if component_names is None:
            component_names = []
        try:
            if not os.path.isdir(self._examples_dir):
                return ToolResponse.success(
                    count=0, examples=[],
                    note="No examples directory found. Build sequences from service knowledge instead.",
                )

            matches = []
            search_terms = set(n.lower() for n in component_names)
            assembly_lower = assembly_name.lower().strip()

            for filename in os.listdir(self._examples_dir):
                if not filename.endswith((".yaml", ".yml")):
                    continue

                filepath = os.path.join(self._examples_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        example = yaml.safe_load(f)
                except Exception:
                    continue

                if example is None:
                    continue

                metadata = example.get("metadata", {})
                example_components = set(
                    c.lower() for c in metadata.get("components", [])
                )
                example_assembly = metadata.get("assembly", "").lower()

                score = 0
                if search_terms and example_components:
                    overlap = search_terms & example_components
                    score = len(overlap) / max(len(search_terms), len(example_components))
                if assembly_lower and assembly_lower in example_assembly:
                    score = max(score, 1.0)

                if score > 0:
                    matches.append({
                        "score": score,
                        "file": filename,
                        "metadata": metadata,
                        "sequence_pattern": example.get("sequence_pattern", []),
                    })

            matches.sort(key=lambda m: -m["score"])

            if not matches:
                return ToolResponse.success(
                    count=0, examples=[],
                    note="No matching examples found. Build the sequence from service knowledge instead.",
                )

            results = []
            for m in matches[:3]:
                results.append({
                    "match_score": round(m["score"], 2),
                    "assembly": m["metadata"].get("assembly", ""),
                    "components": m["metadata"].get("components", []),
                    "task_success": m["metadata"].get("task_success", None),
                    "sequence_pattern": m["sequence_pattern"],
                })

            return ToolResponse.success(count=len(results), examples=results)

        except Exception as e:
            return ToolResponse.error(str(e))

    def _record_knowledge(
        self,
        content: str,
        field: str,
        service_name: str = "",
        category: str = "",
        source: str = "experience",
        confidence: float = 0.7,
    ) -> str:
        """Save new knowledge to the service knowledge base."""
        try:
            if source not in ("user_correction", "experience"):
                return ToolResponse.error(
                    f"Invalid source '{source}'. Must be 'user_correction' or 'experience'."
                )

            valid_fields = ("usage_notes", "parameters")
            if service_name and field not in valid_fields:
                return ToolResponse.error(
                    f"Invalid field '{field}'. Must be one of: {', '.join(valid_fields)}."
                )

            confidence = max(0.0, min(1.0, confidence))

            with self._write_lock:
                data = _load_knowledge(self._knowledge_path)

                if service_name:
                    services = data.setdefault("services", {})

                    if service_name not in services:
                        services[service_name] = {
                            "description": "",
                            "parameters": {},
                            "usage_notes": [],
                        }

                    entry = services[service_name]

                    if field == "parameters":
                        try:
                            param_data = json.loads(content)
                        except json.JSONDecodeError as exc:
                            return ToolResponse.error(
                                f"content must be valid JSON for field='parameters'. "
                                f"Expected: {{\"name\": \"...\", \"type\": \"...\", \"description\": \"...\"}}. "
                                f"Parse error: {exc}"
                            )
                        param_name = param_data.get("name", "").strip()
                        param_type = param_data.get("type", "").strip()
                        param_desc = param_data.get("description", "").strip()
                        if not param_name:
                            return ToolResponse.error(
                                "content JSON must include a non-empty 'name' key."
                            )
                        params_dict = entry.setdefault("parameters", {})
                        params_dict[param_name] = {
                            "type": param_type or "unknown",
                            "description": param_desc,
                        }
                        result_msg = f"Added/updated parameter '{param_name}' in {service_name}."

                    else:
                        target_list = entry.setdefault("usage_notes", [])

                        existing_ids = [n.get("id", "") for n in target_list if isinstance(n, dict)]
                        svc_short = service_name.rsplit("/", 1)[-1] if service_name else "gen"
                        new_id = generate_sequential_id(f"usage_{svc_short}", existing_ids)

                        new_entry = {
                            "id": new_id,
                            "note": content,
                            "confidence": confidence,
                            "source": source,
                            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "confirmation_count": 1,
                            "contradiction_count": 0,
                            "last_confirmed": None,
                            "last_contradicted": None,
                        }
                        target_list.append(new_entry)
                        result_msg = f"Added usage note '{new_id}' to {service_name}."

                else:
                    general = data.setdefault("general_knowledge", [])
                    existing_ids = [g.get("id", "") for g in general if isinstance(g, dict)]
                    new_id = generate_sequential_id("general", existing_ids)

                    new_entry = {
                        "id": new_id,
                        "category": category or "uncategorized",
                        "note": content,
                        "confidence": confidence,
                        "source": source,
                        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    general.append(new_entry)
                    result_msg = f"Added general knowledge entry '{new_id}'."

                _save_knowledge(self._knowledge_path, data)

            self.service_node.get_logger().info(f"KnowledgeTools: {result_msg}")

            return json.dumps({"success": True, "message": result_msg})

        except Exception as e:
            return ToolResponse.error(str(e))

    def _confirm_knowledge(self, service_name: str, note_id: str) -> str:
        """Increment confirmation_count on an existing usage_note."""
        try:
            with self._write_lock:
                data = _load_knowledge(self._knowledge_path)
                entry = data.get("services", {}).get(service_name)
                if entry is None:
                    return ToolResponse.error(
                        f"Service '{service_name}' not found in knowledge base."
                    )
                note = next(
                    (n for n in entry.get("usage_notes", [])
                     if isinstance(n, dict) and n.get("id") == note_id),
                    None,
                )
                if note is None:
                    return ToolResponse.error(
                        f"Note '{note_id}' not found under '{service_name}'."
                    )
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                old_conf = note.get("confidence", 0.6)
                new_conf = min(1.0, round(old_conf + 0.05, 2))
                note["confidence"] = new_conf
                note["confirmation_count"] = note.get("confirmation_count", 0) + 1
                note["last_confirmed"] = now_str
                _save_knowledge(self._knowledge_path, data)

            msg = (
                f"Confirmed '{note_id}' under '{service_name}': "
                f"confirmation_count={note['confirmation_count']}, "
                f"confidence {old_conf} -> {new_conf}."
            )
            self.service_node.get_logger().info(f"KnowledgeTools: {msg}")
            return json.dumps({"success": True, "message": msg})

        except Exception as e:
            return ToolResponse.error(str(e))

    def _contradict_knowledge(self, service_name: str, note_id: str, evidence: str) -> str:
        """Decrement confidence and log evidence on an existing usage_note."""
        try:
            with self._write_lock:
                data = _load_knowledge(self._knowledge_path)
                entry = data.get("services", {}).get(service_name)
                if entry is None:
                    return ToolResponse.error(
                        f"Service '{service_name}' not found in knowledge base."
                    )
                note = next(
                    (n for n in entry.get("usage_notes", [])
                     if isinstance(n, dict) and n.get("id") == note_id),
                    None,
                )
                if note is None:
                    return ToolResponse.error(
                        f"Note '{note_id}' not found under '{service_name}'."
                    )
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
                old_conf = note.get("confidence", 0.6)
                new_conf = max(0.0, round(old_conf - 0.15, 2))
                note["confidence"] = new_conf
                note["contradiction_count"] = note.get("contradiction_count", 0) + 1
                note["last_contradicted"] = now_str
                contradiction_log = note.setdefault("contradiction_notes", [])
                contradiction_log.append(f"[{now_str}] {evidence}")
                _save_knowledge(self._knowledge_path, data)

            msg = (
                f"Contradicted '{note_id}' under '{service_name}': "
                f"contradiction_count={note['contradiction_count']}, "
                f"confidence {old_conf} -> {new_conf}."
            )
            self.service_node.get_logger().info(f"KnowledgeTools: {msg}")
            return json.dumps({"success": True, "message": msg})

        except Exception as e:
            return ToolResponse.error(str(e))
