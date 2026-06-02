"""
Evaluation Pipeline - Includes proper statistical testing and progress logging
"""

import sys
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_all(methods: Dict, X_test: List[str], y_test: List[int],
                 score_methods: Dict = None) -> Dict:
    results = {}
    if score_methods is None:
        score_methods = {}

    n_methods = len(methods)
    for m_idx, (name, predict_fn) in enumerate(methods.items()):
        print(f"  [{m_idx+1}/{n_methods}] Evaluating {name} ...", flush=True)

        preds = []
        for j, t in enumerate(X_test):
            print(f"    Sample {j+1}/{len(X_test)}", end='\r', flush=True)
            preds.append(predict_fn(t))
        print(f"    {name}: {len(preds)} predictions complete.         ")

        # Get scores if available (look in dedicated score_methods dict)
        score_fn = score_methods.get(f"{name}_score", lambda t: 0.0)
        scores = []
        for j, t in enumerate(X_test):
            print(f"    Score {j+1}/{len(X_test)}", end='\r', flush=True)
            scores.append(score_fn(t))
        print(f"    {name}: scores computed.                            ")

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_test, scores)
        except ValueError:
            auc = 0.5

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        n = len(y_test)
        results[name] = {
            "Acc": acc, "Prec": prec, "Rec": rec, "F1": f1, "AUC": auc, "FPR": fpr,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "n_test": n, "n_jb": sum(y_test), "n_benign": len(y_test) - sum(y_test)
        }
    return results


def print_results(results: Dict):
    print("=" * 90)
    print("RESULTS ON REAL-WORLD DATASET")
    print("=" * 90)
    print(f"{'Method':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'FPR':>6} {'n_test':>6}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<20} {r['Acc']:>6.3f} {r['Prec']:>6.3f} {r['Rec']:>6.3f} {r['F1']:>6.3f} {r['AUC']:>6.3f} {r['FPR']:>6.3f} {r['n_test']:>6}")
    print("=" * 90)


def plot_confusion_matrices(results: Dict, save_dir: str = None):
    n_methods = len(results)
    cols = 3
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (name, r) in enumerate(results.items()):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Benign', 'Jailbreak'],
                    yticklabels=['Benign', 'Jailbreak'])
        axes[idx].set_title(f"{name}\n(F1={r['F1']:.3f}, FPR={r['FPR']:.3f})")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(results: Dict, X_test: List[str], y_test: List[int], score_methods: Dict, save_dir: str = None):
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(10, 8))
    for name, r in results.items():
        if f"{name}_score" in score_methods:
            scores = [score_methods[f"{name}_score"](t) for t in X_test]
            fpr, tpr, _ = roc_curve(y_test, scores)
            plt.plot(fpr, tpr, label=f"{name} (AUC={r['AUC']:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Baseline Defenses vs. Adaptive Sentinel")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(f"{save_dir}/roc_curves.png", dpi=150, bbox_inches='tight')
    plt.show()


def category_analysis(sentinel, X_test: List[str], y_test: List[int], categories: List[str]):
    print("\nCategory-Specific Recall (Adaptive Routing):")
    print("-" * 50)
    cat_recalls = {}
    for cat in sorted(set(categories)):
        if cat == "benign":
            continue
        indices = [i for i, c in enumerate(categories) if c == cat]
        cat_labels = [y_test[i] for i in indices]
        cat_preds = []
        for k, i in enumerate(indices):
            print(f"  [{cat}] Predicting {k+1}/{len(indices)} ...", end='\r', flush=True)
            cat_preds.append(sentinel.predict(X_test[i])[0])
        print(f"  [{cat}] Done.                                              ")
        if len(cat_labels) > 0:
            from sklearn.metrics import recall_score
            rec = recall_score(cat_labels, cat_preds, zero_division=0)
            cat_recalls[cat] = rec
            print(f"  {cat:<30}: {rec:.3f}  (n={len(cat_labels)})")
    return cat_recalls


def bootstrap_confidence_intervals(methods: Dict, X_test: List[str], y_test: List[int],
                                    n_bootstrap: int = 1000, confidence: float = 0.95):
    """Compute bootstrap confidence intervals for F1 scores."""
    from sklearn.utils import resample
    from sklearn.metrics import f1_score as _f1

    print(f"\nBootstrap Confidence Intervals ({n_bootstrap} iterations, {confidence*100:.0f}% CI):")
    print("-" * 70)

    for name, predict_fn in methods.items():
        f1_scores = []
        # Progress: announce each method so the log stays live between methods
        print(f"\n  Running bootstrap CI for [{name}] ({n_bootstrap} iterations)...", flush=True)

        for iteration in range(n_bootstrap):
            # Resample test set
            indices = resample(range(len(X_test)), n_samples=len(X_test))
            X_boot = [X_test[i] for i in indices]
            y_boot = [y_test[i] for i in indices]
            preds = [predict_fn(t) for t in X_boot]
            f1 = _f1(y_boot, preds, zero_division=0)
            f1_scores.append(f1)
            print(f"  {name:<25}: Iteration {iteration+1}/{n_bootstrap} - F1={f1:.3f}", end='\r', flush=True)

        f1_scores = np.array(f1_scores)
        alpha = 1 - confidence
        lower = np.percentile(f1_scores, alpha/2 * 100)
        upper = np.percentile(f1_scores, (1 - alpha/2) * 100)
        mean_f1 = np.mean(f1_scores)

        # Print final result on a new line (clears the \r progress line)
        print(f"  {name:<25}: F1 = {mean_f1:.3f} [{lower:.3f}, {upper:.3f}]        ")
