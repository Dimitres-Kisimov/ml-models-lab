"""mllab - a small lab of five from-scratch ML models for a B2B distributor.

Each subpackage implements one published method from scratch, adds a single
principled improvement for the use case, and evaluates with a metric that
survives class imbalance / short series / zeros (not accuracy).

All data is synthetic and seeded; see mllab.synth.
"""

__all__ = ["synth"]
