"""Cost-sensitive fraud detection on PaySim.

Supporting code for *Financial Fraud Detection in Mobile Money Networks Using
Supervised Machine Learning* (COS700). The notebooks orchestrate; the analysis
primitives live here so metric definitions cannot drift between them.

Evaluation is unitless throughout -- a severity ratio and a 0-100 risk score,
never a currency. See ``risk`` for the framework.
"""

from __future__ import annotations

from . import constants, data, evaluate, features, models, risk, splits, viz

__all__ = ["constants", "data", "evaluate", "features", "models", "risk", "splits", "viz"]

__version__ = "0.1.0"
