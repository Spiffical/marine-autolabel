"""MLLM-guided exhaustive instance segmentation of marine life in underwater video.

Two-stage pipeline: a text-prompt SAM3 agent makes a first pass, then a click
engine driven by a vision MLLM recovers what the first pass missed. In practice
the binding constraint is click placement rather than SAM3's masking.
"""

__version__ = "0.1.0"
