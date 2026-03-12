"""
ServiceRegistry: Maps high-level assembly intents to ROS2 service configurations.

This module provides a config-driven abstraction layer between assembly workflow
intents (e.g. ALIGN_GONIO_RIGHT) and the actual ROS2 services that implement them.

Current state: each intent maps 1:1 to one high-level service call.
Future state:  an intent can expand to multiple lower-level service calls
               by updating service_registry.yaml — no code changes needed.
"""

import yaml
import copy
from typing import List, Dict, Any, Optional
from ament_index_python.packages import get_package_share_directory


class ServiceDefinition:
    """Represents a single ROS2 service call within an intent."""

    def __init__(self, client: str, type: str, default_parameters: Dict[str, Any]):
        self.client = client
        self.type = type
        self.default_parameters = default_parameters

    def get_parameters_with_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep-merged copy of default parameters with the provided overrides."""
        merged = copy.deepcopy(self.default_parameters)
        merged.update(overrides)
        return merged

    def __repr__(self) -> str:
        return f"ServiceDefinition(client='{self.client}', type='{self.type}')"


class IntentConfig:
    """Represents an assembly intent with one or more service definitions."""

    def __init__(self, name: str, description: str, services: List[ServiceDefinition]):
        self.name = name
        self.description = description
        self.services = services

    def is_simple(self) -> bool:
        """Returns True when the intent maps to exactly one service call."""
        return len(self.services) == 1

    def __repr__(self) -> str:
        return f"IntentConfig(name='{self.name}', services={len(self.services)})"


class ServiceRegistry:
    """
    Loads service_registry.yaml and provides lookup from intent names
    to their ROS2 service configurations.

    Usage::

        registry = ServiceRegistry()
        intent = registry.get_intent("ALIGN_GONIO_RIGHT")
        for svc in intent.services:
            print(svc.client, svc.default_parameters)
    """

    def __init__(self):
        package_path = get_package_share_directory("pm_co_pilot_planning")
        registry_path = package_path + "/service_registry.yaml"

        with open(registry_path, "r") as f:
            data = yaml.safe_load(f)

        self._intents: Dict[str, IntentConfig] = {}
        for intent_name, intent_data in data.get("intents", {}).items():
            services = [
                ServiceDefinition(
                    client=svc["client"],
                    type=svc.get("type", ""),
                    default_parameters=svc.get("default_parameters", {}) or {},
                )
                for svc in intent_data.get("services", [])
            ]
            self._intents[intent_name] = IntentConfig(
                name=intent_name,
                description=intent_data.get("description", ""),
                services=services,
            )

    def get_intent(self, intent_name: str) -> Optional[IntentConfig]:
        """Return IntentConfig for the given intent name, or None if not found."""
        return self._intents.get(intent_name)

    def get_all_intent_names(self) -> List[str]:
        """Return list of all registered intent names."""
        return list(self._intents.keys())

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return a lightweight summary of all intents for the agent."""
        return [
            {
                "intent": intent.name,
                "description": intent.description,
                "services": [svc.client for svc in intent.services],
            }
            for intent in self._intents.values()
        ]
