"""
Defense Router
Implements BOTH heuristic routing AND learned Random Forest meta-classifier

"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.ensemble import RandomForestClassifier
from .defenses import DefenseMechanism


class DefenseRouter:
    def __init__(self, defenses: List[DefenseMechanism]):
        self.defenses = {d.get_name(): d for d in defenses}
        self.defense_list = defenses
        self.meta_classifier = None
        self.is_trained = False

    def heuristic_route(self, text: str, features: np.ndarray) -> List[Tuple[DefenseMechanism, float]]:
        """Algorithm 1 from paper - heuristic keyword-based routing."""
        text_lower = text.lower()
        weights = {}

        has_override = any(k in text_lower for k in ["ignore", "override", "disregard", "forget", "system prompt"])
        has_encoding = any(k in text_lower for k in ["base64", "rot13", "hex", "decode", "cipher", "encode"])
        has_roleplay = any(k in text_lower for k in ["pretend", "act as", "you are", "DAN", "role", "character"])
        has_framing = any(k in text_lower for k in ["hypothetical", "fictional", "movie", "script", "novel", "scenario", "theoretical"])

        if has_override:
            weights["StructuralPattern"] = weights.get("StructuralPattern", 0) + 0.5
            weights["SemanticDrift"] = weights.get("SemanticDrift", 0) + 0.3
        if has_encoding:
            weights["PerplexityFilter"] = weights.get("PerplexityFilter", 0) + 0.5
            weights["RephrasingStability"] = weights.get("RephrasingStability", 0) + 0.3
        if has_roleplay:
            weights["SemanticDrift"] = weights.get("SemanticDrift", 0) + 0.5
            weights["StructuralPattern"] = weights.get("StructuralPattern", 0) + 0.3
        if has_framing:
            weights["SemanticDrift"] = weights.get("SemanticDrift", 0) + 0.5
            weights["StructuralPattern"] = weights.get("StructuralPattern", 0) + 0.2

        if not weights:
            weights = {"StructuralPattern": 0.35, "SemanticDrift": 0.35, "PerplexityFilter": 0.30}

        total = sum(weights.values())
        selected = [(self.defenses[name], w/total) for name, w in weights.items()]
        return selected

    def train_meta_classifier(self, X_train: List[np.ndarray], y_train: List[int],
                             categories: List[str], defense_scores: Dict[str, List[float]]):
        """Train Random Forest meta-classifier for learned routing.

        Args:
            X_train: List of feature vectors
            y_train: Binary labels (0=benign, 1=jailbreak)
            categories: Attack category labels
            defense_scores: Dict mapping defense name -> list of confidence scores
        """
        n = len(X_train)
        print(f"[DefenseRouter] Building {n}-sample training matrix for Random Forest ...")

        X = []
        y = []
        for i, (features, cat) in enumerate(zip(X_train, categories)):
            print(f"  Building training matrix {i+1}/{n} ...", end='\r', flush=True)
            # Concatenate features with defense scores
            scores = [defense_scores[d.get_name()][i] for d in self.defense_list]
            combined = np.concatenate([features, np.array(scores)])
            X.append(combined)
            y.append(cat)
        print(f"  Training matrix built: {len(X)} × {len(X[0])} dimensions.            ")

        X = np.array(X)
        y = np.array(y)

        print(f"[DefenseRouter] Fitting Random Forest ({len(X)} samples) ...", flush=True)
        self.meta_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.meta_classifier.fit(X, y)
        self.is_trained = True
        print(f"[DefenseRouter] Trained Random Forest on {len(X)} samples")

        # Store feature importances
        importances = self.meta_classifier.feature_importances_
        print(f"[DefenseRouter] Top feature importances: {importances[:5]}")

    def learned_route(self, text: str, features: np.ndarray) -> List[Tuple[DefenseMechanism, float]]:
        """Use trained Random Forest to predict attack category and route accordingly."""
        if not self.is_trained or self.meta_classifier is None:
            return self.heuristic_route(text, features)

        # Get defense scores for this sample
        scores = []
        for d in self.defense_list:
            _, conf = d.analyze(text, features)
            scores.append(conf)

        combined = np.concatenate([features, np.array(scores)]).reshape(1, -1)

        # Predict attack category
        cat_pred = self.meta_classifier.predict(combined)[0]

        # Route based on predicted category
        category_routing = {
            "roleplay": [("SemanticDrift", 0.5), ("StructuralPattern", 0.3)],
            "privilege_escalation": [("StructuralPattern", 0.5), ("SemanticDrift", 0.3)],
            "obfuscation": [("PerplexityFilter", 0.5), ("RephrasingStability", 0.3)],
            "instruction_manipulation": [("StructuralPattern", 0.5), ("SemanticDrift", 0.3)],
            "few_shot": [("StructuralPattern", 0.4), ("RephrasingStability", 0.4)],
            "hypothetical": [("SemanticDrift", 0.5), ("StructuralPattern", 0.2)],
            "benign": [("StructuralPattern", 0.35), ("SemanticDrift", 0.35), ("PerplexityFilter", 0.30)],
        }

        weights = category_routing.get(cat_pred, category_routing["benign"])
        total = sum(w for _, w in weights)
        selected = [(self.defenses[name], w/total) for name, w in weights if name in self.defenses]
        return selected if selected else self.heuristic_route(text, features)

    def full_ensemble_route(self) -> List[Tuple[DefenseMechanism, float]]:
        n = len(self.defense_list)
        return [(d, 1.0/n) for d in self.defense_list]
