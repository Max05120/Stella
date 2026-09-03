"""
registry.py

Central registry of capabilities Stella can actually execute.

mac_knowledge answers:
    "Could macOS do this?"

CapabilityRegistry answers:
    "Can Stella do this right now?"
"""

from core.capabilities.models import Capability


class CapabilityRegistry:

    def __init__(self):
        self._capabilities: dict[
            str,
            Capability
        ] = {}

    def register(
        self,
        capability: Capability,
    ) -> None:

        if capability.name in self._capabilities:
            raise ValueError(
                f"Capability already registered: "
                f"{capability.name}"
            )

        self._capabilities[
            capability.name
        ] = capability

    def get(
        self,
        name: str,
    ) -> Capability | None:

        return self._capabilities.get(name)

    def exists(
        self,
        name: str,
    ) -> bool:

        return name in self._capabilities

    def all(
        self,
    ) -> list[Capability]:

        return list(
            self._capabilities.values()
        )

    def by_family(
        self,
        family: str,
    ) -> list[Capability]:

        return [
            capability
            for capability
            in self._capabilities.values()
            if capability.family == family
        ]

    def names(
        self,
    ) -> list[str]:

        return list(
            self._capabilities.keys()
        )


registry = CapabilityRegistry()