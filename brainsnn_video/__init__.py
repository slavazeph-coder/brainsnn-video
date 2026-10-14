"""Import-safe local BrainSNN Video package; torch loads only on model access."""

from importlib import import_module
from pathlib import Path
import sys

__version__ = "0.1.1"
__all__ = ["VideoPredictor", "ObservationMemory", "ObservationConfig", "MemoryHead",
           "SpectralCoherenceEngine", "StatsEMAEngine"]


def __getattr__(name):
    if name == "VideoPredictor":
        root = Path(__file__).resolve().parent / "engine"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = import_module("wfvideo")
        if Path(module.__file__).resolve() != root / "wfvideo.py":
            raise RuntimeError("wfvideo is shadowed; import this checkout in a fresh Python process")
        return module.VideoPredictor
    if name in ("ObservationMemory", "ObservationConfig"):
        module = import_module(".engine.observation_memory_v2.observation", __name__)
        return getattr(module, name)
    if name == "MemoryHead":
        module = import_module(".engine.observation_memory_v2.adapters", __name__)
        return module.MemoryHead
    if name in ("SpectralCoherenceEngine", "StatsEMAEngine"):
        root = Path(__file__).resolve().parent / "engine"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        module = import_module("crumb_coherence")
        if Path(module.__file__).resolve() != root / "crumb_coherence" / "__init__.py":
            raise RuntimeError("analytic engine is shadowed; use this checkout in a fresh Python process")
        return getattr(module, name)
    raise AttributeError(name)
