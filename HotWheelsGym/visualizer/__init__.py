"""Optional web debugger for the 67-D RAM observation contract.

Importing this package does not import or start the HTTP server.  The adapter and
snapshot models have no web-framework dependencies.
"""

from .adapter import ObservationVisualizerAdapter, build_visualizer_snapshot
from .models import VisualizerSnapshot

__all__ = [
    "ObservationVisualizerAdapter",
    "VisualizerSnapshot",
    "build_visualizer_snapshot",
]
