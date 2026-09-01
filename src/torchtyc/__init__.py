"""torchtyc: shape checking for PyTorch, from your jaxtyping annotations."""

from .config import Config, load
from .diagnostics import RULES, Diagnostic, Rule, Severity
from .engine import Report, check_paths, collect_files

__version__ = "0.1.2"

__all__ = [
    "RULES",
    "Config",
    "Diagnostic",
    "Report",
    "Rule",
    "Severity",
    "__version__",
    "check_paths",
    "collect_files",
    "load",
]
