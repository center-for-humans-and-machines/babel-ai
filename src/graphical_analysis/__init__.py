"""Deprecated CSV plotting helpers — use ``persistence`` and ``viz``."""

import warnings

warnings.warn(
    "graphical_analysis is deprecated; use persistence.run_store and viz",
    DeprecationWarning,
    stacklevel=2,
)
