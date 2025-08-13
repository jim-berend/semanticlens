"""
Component visualization modules for semantic analysis.

This module provides different approaches for visualizing and analyzing
neural network components, including activation-based and relevance-based
visualization methods.

Classes
-------
ActivationComponentVisualizer
    Visualizer using activation maximization techniques.
RelevanceComponentVisualizer
    Visualizer using relevance-based attribution methods.
"""

from semanticlens.component_visualization.activation_based import ActivationComponentVisualizer
from semanticlens.component_visualization.relevance_based import RelevanceComponentVisualizer

__all__ = [
    "ActivationComponentVisualizer",
    "RelevanceComponentVisualizer",
]
