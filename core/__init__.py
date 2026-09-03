"""
core.capabilities

Loading this package registers Stella's built-in capabilities.
"""

from core.capabilities.registry import registry

# Import capability modules so their registration code runs.
# from core.capabilities.workspace import workspace  # noqa: F401

__all__ = [
    "registry",
]