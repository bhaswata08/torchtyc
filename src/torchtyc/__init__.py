"""torchtyc: static array shape checking for PyTorch, powered by meta tensors."""

from .config import Config, load
from .diagnostics import RULES, Diagnostic, Rule, Severity
from .engine import Report, check_paths, collect_files

__version__ = "0.1.0"

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
