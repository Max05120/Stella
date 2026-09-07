"""
Stella Mac capability system.
"""

from core.capabilities.workspace import (
    register_workspace_capabilities,
)
from core.capabilities.filesystem import (
    register_filesystem_capabilities,
)
from core.capabilities.discovery import (
    register_discovery_capabilities,
)
from core.capabilities.finder import (
    register_finder_capabilities,
)
from core.capabilities.system import (
    register_system_capabilities,
)
from core.capabilities.applications import (
    register_application_capabilities,
)

from core.capabilities.windows import (
    register_window_capabilities,
)
from core.capabilities.accessibility import (
    register_accessibility_capabilities,
)

_initialized = False


def initialize_capabilities():
    global _initialized

    if _initialized:
        return

    register_workspace_capabilities()
    register_filesystem_capabilities()
    register_discovery_capabilities()
    register_finder_capabilities()
    register_system_capabilities()
    register_application_capabilities()
    register_window_capabilities()
    register_accessibility_capabilities()
    _initialized = True