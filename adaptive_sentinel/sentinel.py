"""
Adaptive Sentinel - MAIN PIPELINE
Three-stage: Feature Extraction -> Defense Routing -> Ensemble Decision
Now with REAL embeddings, REAL perplexity, REAL rephrasing, and learned components

"""

import sys
from typing import Tuple, List, Dict
from .feature_extractor import FeatureExtractor
from .defenses import PerplexityFilter, SemanticDrift, StructuralPattern, RephrasingStability
from .router import DefenseRouter
from .ensemble import WeightedEnsemble


class AdaptiveSentinel:
    def __init__(self, mode: str = "adaptive", threshold: float = 0.5,
                 use_real_embeddings: bool = True, use_real_perplexity: bool = True,
                 use_real_paraphrase: bool = True, use_learned_router: bool = False,
                 use_gradient_boosting: bool = False):
        self.mode = mode
        self.threshold = threshold
        self.use_learned_router = use_learned_router
        self.use_gradient_boosting = use_gradient_boosting

        self.fe = FeatureExtractor(
            use_real_embeddings=use_real_embeddings,
            use_real_perplexity=use_real_perplexity
        )
        self.defenses = [
            PerplexityFilter(),
            SemanticDrift(),
            StructuralPattern(),
            RephrasingStability(use_real_paraphrase=use_real_paraphrase)
        ]
        self.router = DefenseRouter(self.defenses)
        self.ensemble = WeightedEnsemble(threshold, use_gradient_boosting=use_gradient_boosting)

    def fit(self, X_train: List[str], y_train: List[int], categories: List[str]):
        """Fit benign reference and train learned components.
        """
        n = len(X_train)

        # ------------------------------------------------------------------ #
        # Step 1 – fit benign reference centroid BEFORE extracting features   #
        # (previously this was done AFTER, making feature[41] wrong for all   #
        # training samples because _benign_centroid was None at extract time)  #
        # ------------------------------------------------------------------ #
        benign_texts = [t for t, y in zip(X_train, y_train) if y == 0]
        if benign_texts:
            self.fe.fit_benign_reference(benign_texts)

        # ------------------------------------------------------------------ #
        # Step 2 – extract features for all training samples                   #
        # ------------------------------------------------------------------ #
        print(f"[AdaptiveSentinel] Extracting features for {n} training samples ...")
        train_features = []
        for i, t in enumerate(X_train):
            print(f"  Feature extraction {i+1}/{n} ({(i+1)/n*100:.0f}%)", end='\r', flush=True)
            train_features.append(self.fe.extract(t))
        print(f"  Feature extraction complete: {n} samples.                          ")

        # ------------------------------------------------------------------ #
        # Step 3 – compute defense confidence scores for every training sample #
        # ------------------------------------------------------------------ #
        defense_scores = {d.get_name(): [] for d in self.defenses}
        print(f"[AdaptiveSentinel] Computing defense scores for {n} training samples ...")
        for i, (text, features) in enumerate(zip(X_train, train_features)):
            print(f"  Defense scoring {i+1}/{n} ({(i+1)/n*100:.0f}%)", end='\r', flush=True)
            for d in self.defenses:
                _, conf = d.analyze(text, features)
                defense_scores[d.get_name()].append(conf)
        print(f"  Defense scoring complete.                                             ")

        # ------------------------------------------------------------------ #
        # Step 4 – train learned router if requested                           #
        # ------------------------------------------------------------------ #
        if self.use_learned_router:
            self.router.train_meta_classifier(train_features, y_train, categories, defense_scores)

        # ------------------------------------------------------------------ #
        # Step 5 – train Gradient Boosting if requested                        #
        # ------------------------------------------------------------------ #
        if self.use_gradient_boosting:
            all_outputs: Dict[str, List] = {d.get_name(): [] for d in self.defenses}
            print(f"[AdaptiveSentinel] Building defense outputs for Gradient Boosting ({n} samples) ...")
            for i, (t, feat) in enumerate(zip(X_train, train_features)):
                print(f"  GB output {i+1}/{n} ({(i+1)/n*100:.0f}%)", end='\r', flush=True)
                for d in self.defenses:
                    flag, conf = d.analyze(t, feat)
                    all_outputs[d.get_name()].append((flag, conf))
            print(f"  GB defense outputs complete.                                       ")
            self.ensemble.train_gradient_boosting(train_features, all_outputs, y_train)

        print(f"[AdaptiveSentinel] Fitted on {n} samples")
        return self

    def predict(self, text: str) -> Tuple[int, float, List[str]]:
        features = self.fe.extract(text)

        if self.mode == "adaptive":
            if self.use_learned_router and self.router.is_trained:
                selected = self.router.learned_route(text, features)
            else:
                selected = self.router.heuristic_route(text, features)
        elif self.mode == "full":
            selected = self.router.full_ensemble_route()
        else:
            selected = self.router.heuristic_route(text, features)

        # Get all defense outputs for gradient boosting
        all_outputs = {}
        for d in self.defenses:
            flag, conf = d.analyze(text, features)
            all_outputs[d.get_name()] = (flag, conf)

        pred, score = self.ensemble.decide(selected, text, features, all_outputs)
        defense_names = [d.get_name() for d, _ in selected]
        return pred, score, defense_names

    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float, List[str]]]:
        results = []
        n = len(texts)
        for i, t in enumerate(texts):
            print(f"  [PredictBatch] {i+1}/{n} ...", end='\r', flush=True)
            results.append(self.predict(t))
        print(f"  [PredictBatch] Complete: {n} predictions.            ")
        return results
