#!/usr/bin/env python3
"""
extract_knowledge.py — Offline script to process copilot interaction logs
into example assembly patterns for the knowledge base.

Scans RSAP_Processes/ (recursively) for copilot_log_*.json files with
task_success == true, extracts the final sequence into a simplified
example YAML, and writes it to Knowledge_Base/examples/.

Usage:
    python3 extract_knowledge.py [--db-path /path/to/co_pilot_planning_test]

If --db-path is not provided, reads from assembly_config.yaml in the
pm_co_pilot_planning ROS package share directory.
"""

import argparse
import json
import os
import re
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional


def find_log_files(rsap_root: str) -> List[str]:
    """Recursively find all copilot_log_*.json files."""
    results = []
    for dirpath, _, filenames in os.walk(rsap_root):
        for fn in filenames:
            if fn.startswith("copilot_log_") and fn.endswith(".json"):
                results.append(os.path.join(dirpath, fn))
    return sorted(results)


def extract_components_from_sequence(actions: List[Dict]) -> List[str]:
    """Extract unique component names from spawn actions in a sequence."""
    components = []
    for action in actions:
        client = action.get("client", "")
        params = action.get("parameters", {})

        # Look for spawn_object actions
        if "spawn" in client.lower():
            # Component name might be in various parameter fields
            for key in ("object_name", "component_name", "name"):
                val = params.get(key, "")
                if val and val not in components:
                    components.append(val)
                    break
    return components


def extract_assembly_name(actions: List[Dict]) -> str:
    """Try to extract assembly name from assembly instruction actions."""
    for action in actions:
        client = action.get("client", "")
        params = action.get("parameters", {})
        if "assembly" in client.lower() and "instruction" in client.lower():
            for key in ("assembly_name", "name", "file_path"):
                val = params.get(key, "")
                if val:
                    return os.path.basename(val).replace(".json", "")
    return ""


def simplify_action(action: Dict) -> Dict[str, Any]:
    """Convert a raw action from the log into a simplified step."""
    client = action.get("client", "")
    name = action.get("name", client)
    params = action.get("parameters", {})

    step = {"action": name, "service": client}

    # Extract key parameters based on service type
    for key in ("frame_name", "object_name", "component_name", "file_path",
                "camera_name", "process_name", "gripper_tip", "start_frame",
                "target_frame"):
        if key in params and params[key]:
            step[key] = params[key]

    return step


def process_log(log_path: str) -> Optional[Dict[str, Any]]:
    """Process a single log file into an example pattern."""
    with open(log_path, "r") as f:
        log_data = json.load(f)

    # Only process successful sessions
    if not log_data.get("task_success"):
        return None

    final_seq = log_data.get("final_sequence", {})
    actions = final_seq.get("actions", [])

    if not actions:
        return None

    components = extract_components_from_sequence(actions)
    assembly_name = extract_assembly_name(actions)

    # Build simplified sequence pattern
    sequence_pattern = []
    for idx, action in enumerate(actions, 1):
        step = simplify_action(action)
        step["step"] = idx
        sequence_pattern.append(step)

    # Build assembly description
    assembly_desc = assembly_name if assembly_name else " + ".join(components)

    example = {
        "metadata": {
            "assembly": assembly_desc,
            "components": components,
            "task_success": True,
            "source_log": os.path.basename(log_path),
            "model": log_data.get("model", "unknown"),
            "extracted_date": datetime.now().isoformat(),
            "session_summary": log_data.get("summary", {}),
        },
        "sequence_pattern": sequence_pattern,
    }

    return example


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge from copilot logs")
    parser.add_argument("--db-path", type=str, default=None,
                        help="Path to the assembly database root")
    args = parser.parse_args()

    # Resolve database path
    db_root = args.db_path
    if db_root is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            pkg_path = get_package_share_directory("pm_co_pilot_planning")
            config_path = os.path.join(pkg_path, "assembly_config.yaml")
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            db_root = cfg.get("assembly_database_path", "")
        except Exception:
            db_root = "/home/match-pm/Documents/co_pilot_planning_test"
            print(f"Warning: Could not load ROS config, using default: {db_root}")

    rsap_root = os.path.join(db_root, "RSAP_Processes")
    examples_dir = os.path.join(db_root, "Knowledge_Base", "examples")
    os.makedirs(examples_dir, exist_ok=True)

    # Find already-processed logs
    existing = set()
    for fn in os.listdir(examples_dir):
        if fn.endswith((".yaml", ".yml")):
            filepath = os.path.join(examples_dir, fn)
            try:
                with open(filepath, "r") as f:
                    data = yaml.safe_load(f)
                source_log = data.get("metadata", {}).get("source_log", "")
                if source_log:
                    existing.add(source_log)
            except Exception:
                pass

    # Process new logs
    log_files = find_log_files(rsap_root)
    print(f"Found {len(log_files)} log files in {rsap_root}")

    new_count = 0
    skip_count = 0

    for log_path in log_files:
        log_name = os.path.basename(log_path)
        if log_name in existing:
            skip_count += 1
            continue

        try:
            example = process_log(log_path)
        except Exception as e:
            print(f"  Error processing {log_name}: {e}")
            continue

        if example is None:
            continue

        # Generate output filename
        components = example["metadata"]["components"]
        if components:
            safe_name = "_".join(c.lower() for c in components[:3])
        else:
            safe_name = "unknown"
        # Add timestamp suffix to avoid collisions
        ts = re.search(r"(\d{8}_\d{6})", log_name)
        ts_str = ts.group(1) if ts else datetime.now().strftime("%Y%m%d_%H%M%S")
        out_filename = f"{safe_name}_{ts_str}.yaml"
        out_path = os.path.join(examples_dir, out_filename)

        with open(out_path, "w") as f:
            yaml.dump(example, f, default_flow_style=False, sort_keys=False)

        print(f"  Extracted: {log_name} → {out_filename}")
        new_count += 1

    print(f"\nDone. Extracted {new_count} new examples, skipped {skip_count} already processed.")


if __name__ == "__main__":
    main()
