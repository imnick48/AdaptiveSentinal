"""
Weighted Ensemble Decision Layer
Includes both heuristic weighting AND Gradient Boosting classifier

"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from .defenses import DefenseMechanism


class WeightedEnsemble:
    def __init__(self, threshold: float = 0.5, use_gradient_boosting: bool = False):
        self.threshold = threshold
        self.use_gradient_boosting = use_gradient_boosting
        self.gb_model = None
        self.is_trained = False

    def train_gradient_boosting(self, X_features: List[np.ndarray], 
                                 defense_outputs: Dict[str, List[Tuple[int, float]]],
                                 y_train: List[int]):
        """Train Gradient Boosting on concatenated features + defense outputs."""
        if not self.use_gradient_boosting:
            return

        X = []
        for i in range(len(X_features)):
            features = X_features[i]
            # Concatenate with defense outputs
            outputs = []
            for defense_name in ["PerplexityFilter", "SemanticDrift", "StructuralPattern", "RephrasingStability"]:
                if defense_name in defense_outputs:
                    flag, conf = defense_outputs[defense_name][i]
                    outputs.extend([flag, conf])
                else:
                    outputs.extend([0, 0.0])
            combined = np.concatenate([features, np.array(outputs)])
            X.append(combined)

        X = np.array(X)
        y = np.array(y_train)

        print(f"[WeightedEnsemble] Fitting Gradient Boosting ({len(X)} samples) ...", flush=True)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(X, y)
        self.is_trained = True
        print(f"[WeightedEnsemble] Trained Gradient Boosting on {len(X)} samples")

    def decide(self, selected_defenses: List[Tuple[DefenseMechanism, float]], 
               text: str, features: np.ndarray,
               all_defense_outputs: Dict[str, Tuple[int, float]] = None) -> Tuple[int, float]:
        """Returns (prediction, confidence_score)."""

        # Use Gradient Boosting if trained and available
        if self.use_gradient_boosting and self.is_trained and self.gb_model is not None and all_defense_outputs is not None:
            return self._gb_decide(features, all_defense_outputs)

        # Fallback to heuristic weighted voting (paper Equation 5)
        numerator = 0
        denominator = 0

        for defense, weight in selected_defenses:
            s_i, c_i = defense.analyze(text, features)
            defense_score = 0.7 * s_i + 0.3 * c_i
            numerator += weight * defense_score
            denominator += weight

        if denominator == 0:
            return 0, 0.0

        score = numerator / denominator
        prediction = 1 if score > self.threshold else 0
        return prediction, score

    def _gb_decide(self, features: np.ndarray, all_defense_outputs: Dict[str, Tuple[int, float]]) -> Tuple[int, float]:
        """Use trained Gradient Boosting for decision."""
        outputs = []
        for defense_name in ["PerplexityFilter", "SemanticDrift", "StructuralPattern", "RephrasingStability"]:
            if defense_name in all_defense_outputs:
                flag, conf = all_defense_outputs[defense_name]
                outputs.extend([flag, conf])
            else:
                outputs.extend([0, 0.0])

        combined = np.concatenate([features, np.array(outputs)]).reshape(1, -1)
        prob = self.gb_model.predict_proba(combined)[0]
        score = prob[1]  # Probability of jailbreak class
        prediction = 1 if score > self.threshold else 0
        return prediction, score
