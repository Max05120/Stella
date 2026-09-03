"""
Stella Mac capability system.
"""

from core.capabilities.workspace import (
    register_workspace_capabilities,
)


_initialized = False


def initialize_capabilities():
    global _initialized

    if _initialized:
        return

    register_workspace_capabilities()

    _initialized = True