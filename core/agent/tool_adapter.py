"""
core/agent/tool_adapter.py

Safe bridge between Stella's agent runtime and the existing 
capability system.

The agent must never call capability handlers directly.

All execution continues through execute_capability().
"""

from __future__ import annotations

import inspect

from dataclasses import dataclass, field
from typing import Any

from core.agent.models import AgentObservation
from core.agent.observer import observe_execution

from core.capabilities import (
    initialize_capabilities,
)

from core.capabilities.executor import (
    execute_capability,
)

from core.capabilities.registry import (
    registry,
)

@dataclass(slots=True)
class AgentToolParameter:
    """
    One argument accepted by a capability handler.
    """

    name: str
    type_name: str
    required: bool
    default: Any = None


@dataclass(slots=True)
class AgentTool:
    """
    Agent-facing description of one executable Stella capability.
    """

    name: str
    family: str
    description: str

    risk: str

    requires_confirmation: bool
    supports_undo: bool

    permissions: list[str]

    parameters: list[AgentToolParameter] = field(
        default_factory=list
    )


class AgentToolAdapter:
    """
    Controlled interface through which the agent sees and executes tools.

    The planner interacts with this clss rather than directly accessing
    CapabilityRegistry or capability handlers.
    """

    def __init__(self):
        initialize_capabilities()

    def list_tools(self) -> list[AgentTool]:
        """
            Return all registered capabilities with their real handler schemes.
        """

        return [
            self._capability_to_agent_tool(
                capability
            )
            for capability in registry.all()
        ]
    
    def get_tool(
            self,
            name: str,
    ) -> AgentTool | None:
        """
        Return one capability's metadata and argument schema.
        """

        capability = registry.get(name)

        if capability is None:
            return None
        
        return self._capability_to_agent_tool(
            capability
        )
    
    def validate_arguments(
            self,
            capability_name: str,
            arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """
        Validate a planner-generated argument dictionary against
        the real capability handler signature.

        This performs validation WITHOUT executing anything
        """

        capability = registry.get(
            capability_name
        )

        if capability is None:
            return (
                False,
                (
                    "Capability is not registered: "
                    f"{capability_name}"
                ),
            )
        
        try:
            signature = inspect.signature(
                capability.handler
            )

            signature.bind(
                **arguments
            )
        
        except TypeError as exc:
            return (
                False,
                str(exc),
            )
        return (
            True,
            None,
        )
    
    def execute(
            self,
            capability_name: str,
            arguments: dict[str, Any] | None = None,
            *,
            confirmed: bool = False,
    ) -> AgentObservation:
        """
        Execute through Stella's existing controlled executor.
        """

        capability = registry.get(
            capability_name
        )

        requires_confirmation = bool(
            capability
            and capability.requires_confirmation
        )

        execution = execute_capability(
            capability_name,
            arguments or {},
            confirmed=confirmed,
        )

        return observe_execution(
            execution,
            requires_confirmation=(
                requires_confirmation
            ),
        )
    
    def _capability_to_agent_tool(
            self,
            capability,
    ) -> AgentTool:
        """
        Convert a registered Capability into the safe agent-facing model.
        """

        risk = getattr(
            capability.risk,
            "value",
            capability.risk,
        )

        return AgentTool(
            name=capability.name,
            family=capability.family,
            description=capability.description,
            risk=str(risk),
            requires_confirmation=(
                capability.requires_confirmation
            ),
            supports_undo=(
                capability.supports_undo
            ),
            permissions=list(
                capability.permissions
            ),
            parameters=(
                self._extract_parameters(
                    capability.handler
                )
            ),
        )
    
    def _extract_parameters(
            self,
            handler,
    ) -> list[AgentToolParameter]:
        """
        Inspect the real Python Handler Signature.

        Example:

            def create_folder(
            parent_path: str,
            folder_name: str,
            )
        becomes:

            parent_path: str required
            folder_name: str required
        """

        signature = inspect.signature(
            handler
        )

        parameters: list[
            AgentToolParameter
        ] = []

        for parameter in (
            signature.parameters.values()
        ):
            # Skip *args / **kwargs style parameters.
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue

            annotation = (
                parameter.annotation
            )

            if (
                annotation
                is inspect.Parameter.empty
            ):
                type_name = "any"
            else:
                type_name = self._type_name(
                    annotation
                )
            
            required = (
                parameter.default
                is inspect.Parameter.empty
            )

            default = (
                None
                if required
                else parameter.default
            )
            parameters.append(
                AgentToolParameter(
                    name=parameter.name,
                    type_name=type_name,
                    required=required,
                    default=default,
                )
            )
        
        return parameters
    
    @staticmethod
    def _type_name(
        annotation,
    ) -> str:
        """
        Produce a compact readable type name for prompts.
        """

        if isinstance(
            annotation,
            str,
        ):
            return annotation
        
        name = getattr(
            annotation,
            "__name__",
            None,
        )
        if name: return name

        return str(annotation).replace(
            "typing.",
            "",
        )