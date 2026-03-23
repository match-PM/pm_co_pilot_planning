"""
KnowledgeTools: LangChain tools for autonomous domain knowledge management.

Provides three tools:
  - query_assembly_knowledge: Retrieve rules by category/tags from the knowledge base
  - get_similar_assembly_example: Find reference execution traces by component/assembly name
  - record_knowledge: Save new rules learned from experience or user corrections
"""

import json
import os
import yaml
from datetime import date
from typing import Optional, List, Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryKnowledgeInput(BaseModel):
    pass


class GetExampleInput(BaseModel):
    component_names: List[str] = Field(
        default_factory=list,
        description="List of component names to search for in example traces.",
    )
    assembly_name: str = Field(
        default="",
        description="Assembly name to search for. If provided, matches examples for this assembly.",
    )


class RecordKnowledgeInput(BaseModel):
    rule: str = Field(
        description="The rule or knowledge to record, as a clear imperative statement.",
    )
    category: str = Field(
        description=(
            "Category for this rule: ordering, tool_usage, component_role, "
            "parallelization, error_recovery, or a new category if needed."
        ),
    )
    tags: List[str] = Field(
        description="Tags for retrieval. Use descriptive terms related to the rule.",
    )
    source: str = Field(
        description=(
            "Source of this knowledge: 'user_correction' (user explicitly corrected you) "
            "or 'experience' (learned from execution feedback or discovered pattern)."
        ),
    )
    confidence: float = Field(
        default=0.7,
        description=(
            "Confidence level 0.0-1.0. User corrections: 0.9, resolved errors: 0.7, "
            "discovered patterns: 0.6. Rules below 0.7 are flagged for review."
        ),
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


def _load_rules(rules_path: str) -> Dict[str, Any]:
    """Load the rules.yaml file. Returns empty structure if not found."""
    if not os.path.isfile(rules_path):
        return {"metadata": {"version": 1}, "rules": []}
    with open(rules_path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {"metadata": {"version": 1}, "rules": []}
    return data


def _save_rules(rules_path: str, data: Dict[str, Any]) -> None:
    """Save the rules data back to rules.yaml."""
    data["metadata"]["last_updated"] = date.today().isoformat()
    with open(rules_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


_SOURCE_PRIORITY = {"user_correction": 3, "experience": 2, "bootstrap": 1}


def _rule_sort_key(rule: Dict[str, Any]):
    """Sort rules by source priority (desc) then confidence (desc)."""
    return (
        -_SOURCE_PRIORITY.get(rule.get("source", ""), 0),
        -rule.get("confidence", 0.0),
    )


# ---------------------------------------------------------------------------
# KnowledgeTools class
# ---------------------------------------------------------------------------

class KnowledgeTools:
    """
    Provides LangChain tools for querying and updating the domain knowledge base.
    """

    def __init__(self, service_node: Node):
        self.service_node = service_node
        self.service_node.get_logger().info("Initializing KnowledgeTools...")

        try:
            cfg = _load_assembly_config()
            self._db_root = cfg.get("assembly_database_path", "")
            kb_subdir = cfg.get("knowledge_base_subdir", "Knowledge_Base")
            self._kb_root = os.path.join(self._db_root, kb_subdir)
            self._rules_path = os.path.join(self._kb_root, "rules.yaml")
            self._examples_dir = os.path.join(self._kb_root, "examples")
        except Exception as e:
            self.service_node.get_logger().warning(
                f"KnowledgeTools: could not load assembly_config.yaml: {e}"
            )
            self._db_root = ""
            self._kb_root = ""
            self._rules_path = ""
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
                "Return ALL domain knowledge rules (ordering constraints, tool usage "
                "patterns, component roles, parallelization, error recovery). "
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
                "Save a new domain rule to the knowledge base for future sessions. "
                "Call this when:\n"
                "  - The user corrects your approach (source='user_correction', confidence=0.9)\n"
                "  - You resolve an execution error and learn a lesson (source='experience', confidence=0.7)\n"
                "  - You discover a working pattern (source='experience', confidence=0.6)\n\n"
                "Do NOT record facts discoverable via existing tools (component frames, "
                "service parameters). Only record procedural knowledge and rules."
            ),
            args_schema=RecordKnowledgeInput,
        )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    def _query_assembly_knowledge(self) -> str:
        """Return all non-superseded rules sorted by authority and confidence."""
        try:
            data = _load_rules(self._rules_path)
            rules = data.get("rules", [])

            # Filter out superseded rules
            rules = [r for r in rules if "superseded_by" not in r]

            # Sort by authority and confidence
            rules.sort(key=_rule_sort_key)

            # Format output
            if not rules:
                return json.dumps({
                    "success": True,
                    "count": 0,
                    "rules": [],
                    "note": "No matching rules found. You may need to reason from first principles or ask the user.",
                })

            formatted = []
            for r in rules:
                entry = {
                    "id": r.get("id", ""),
                    "rule": r.get("rule", ""),
                    "category": r.get("category", ""),
                    "confidence": r.get("confidence", 0.0),
                    "source": r.get("source", ""),
                }
                if r.get("confidence", 1.0) < 0.7:
                    entry["review_flag"] = "Low confidence - verify before relying on this rule"
                formatted.append(entry)

            return json.dumps({
                "success": True,
                "count": len(formatted),
                "rules": formatted,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _get_similar_assembly_example(
        self, component_names: List[str] = None, assembly_name: str = ""
    ) -> str:
        """Search for reference execution traces matching component or assembly names."""
        if component_names is None:
            component_names = []
        try:
            if not os.path.isdir(self._examples_dir):
                return json.dumps({
                    "success": True,
                    "count": 0,
                    "examples": [],
                    "note": "No examples directory found. Build sequences from rules instead.",
                })

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

                # Score by overlap
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

            # Sort by score descending
            matches.sort(key=lambda m: -m["score"])

            if not matches:
                return json.dumps({
                    "success": True,
                    "count": 0,
                    "examples": [],
                    "note": "No matching examples found. Build the sequence from rules instead.",
                })

            # Return top 3
            results = []
            for m in matches[:3]:
                results.append({
                    "match_score": round(m["score"], 2),
                    "assembly": m["metadata"].get("assembly", ""),
                    "components": m["metadata"].get("components", []),
                    "task_success": m["metadata"].get("task_success", None),
                    "sequence_pattern": m["sequence_pattern"],
                })

            return json.dumps({
                "success": True,
                "count": len(results),
                "examples": results,
            })

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})

    def _record_knowledge(
        self,
        rule: str,
        category: str,
        tags: List[str],
        source: str,
        confidence: float = 0.7,
    ) -> str:
        """Save a new rule to the knowledge base."""
        try:
            # Validate source
            if source not in ("user_correction", "experience"):
                return json.dumps({
                    "success": False,
                    "error": f"Invalid source '{source}'. Must be 'user_correction' or 'experience'.",
                })

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            data = _load_rules(self._rules_path)
            existing_rules = data.get("rules", [])

            # Generate next ID for this category
            category_rules = [r for r in existing_rules if r.get("category") == category]
            max_num = 0
            for r in category_rules:
                rid = r.get("id", "")
                parts = rid.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        max_num = max(max_num, int(parts[1]))
                    except ValueError:
                        pass
            new_id = f"{category}_{max_num + 1:03d}"

            # Check for potentially contradictory rules (same category + overlapping tags)
            new_tag_set = set(t.lower() for t in tags)
            for existing in existing_rules:
                if existing.get("category") != category:
                    continue
                if "superseded_by" in existing:
                    continue
                existing_tag_set = set(t.lower() for t in existing.get("tags", []))
                overlap = new_tag_set & existing_tag_set
                # If >50% tag overlap and new source has higher authority, mark old as superseded
                if existing_tag_set and len(overlap) / len(existing_tag_set) > 0.5:
                    new_priority = _SOURCE_PRIORITY.get(source, 0)
                    old_priority = _SOURCE_PRIORITY.get(existing.get("source", ""), 0)
                    if new_priority > old_priority:
                        existing["superseded_by"] = new_id
                        self.service_node.get_logger().info(
                            f"KnowledgeTools: Rule '{existing.get('id')}' superseded by '{new_id}'"
                        )

            # Create new rule
            new_rule = {
                "id": new_id,
                "category": category,
                "rule": rule,
                "confidence": confidence,
                "source": source,
                "tags": tags,
                "created": date.today().isoformat(),
                "last_validated": date.today().isoformat(),
            }

            existing_rules.append(new_rule)
            data["rules"] = existing_rules
            _save_rules(self._rules_path, data)

            self.service_node.get_logger().info(
                f"KnowledgeTools: Recorded rule '{new_id}': {rule[:80]}..."
            )

            result = {
                "success": True,
                "rule_id": new_id,
                "message": f"Rule '{new_id}' saved to knowledge base.",
            }
            if confidence < 0.7:
                result["review_flag"] = "Low confidence - this rule will be flagged for human review."

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"success": False, "error": str(e)})
