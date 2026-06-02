"""
Adaptive Sentinel - Implementation
Uses real SBERT embeddings, real GPT-2 perplexity, real T5 paraphrasing,
learned Random Forest router, and Gradient Boosting ensemble.
"""

__version__ = "1.0.0"
__author__ = "Sagnick Das"

from .feature_extractor import FeatureExtractor
from .defenses import PerplexityFilter, SemanticDrift, StructuralPattern, RephrasingStability
from .router import DefenseRouter
from .ensemble import WeightedEnsemble
from .sentinel import AdaptiveSentinel
from .dataset import load_dataset, split_dataset
from .evaluate import evaluate_all, print_results, plot_confusion_matrices, plot_roc_curves, category_analysis, bootstrap_confidence_intervals

__all__ = [
    "FeatureExtractor",
    "PerplexityFilter", "SemanticDrift", "StructuralPattern", "RephrasingStability",
    "DefenseRouter",
    "WeightedEnsemble",
    "AdaptiveSentinel",
    "load_dataset", "split_dataset",
    "evaluate_all", "print_results", "plot_confusion_matrices", "plot_roc_curves",
    "category_analysis", "bootstrap_confidence_intervals"
]
