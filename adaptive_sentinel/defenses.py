"""
Individual Defense Mechanisms 
- PerplexityFilter: REAL GPT-2 perplexity detection
- SemanticDrift: REAL SBERT embedding drift
- StructuralPattern: Regex-based template detection
- RephrasingStability: REAL T5 paraphrase model
"""

import numpy as np
import re
from typing import Tuple, List
from .feature_extractor import FeatureExtractor


class DefenseMechanism:
    def analyze(self, text: str, features: np.ndarray) -> Tuple[int, float]:
        raise NotImplementedError

    def get_name(self) -> str:
        raise NotImplementedError

    def get_specialization(self) -> List[str]:
        raise NotImplementedError


class PerplexityFilter(DefenseMechanism):
    """Detects high-perplexity patterns using REAL GPT-2 perplexity."""

    def analyze(self, text: str, features: np.ndarray) -> Tuple[int, float]:
        # features[42] is REAL perplexity from GPT-2
        ppl = features[42] if len(features) > 42 else 0
        # Threshold calibrated on training data
        flag = 1 if ppl > 0.35 else 0  # Adjusted for normalized perplexity
        confidence = min(ppl / 0.5, 1.0)
        return flag, confidence

    def get_name(self): 
        return "PerplexityFilter"

    def get_specialization(self): 
        return ["obfuscation", "few_shot"]


class SemanticDrift(DefenseMechanism):
    """Detects semantic deviation using REAL SBERT embeddings."""

    def analyze(self, text: str, features: np.ndarray) -> Tuple[int, float]:
        # features[41] is REAL embedding drift from SBERT
        drift = features[41] if len(features) > 41 else 0
        # Threshold: higher drift = more suspicious
        flag = 1 if drift > 0.15 else 0  # Adjusted for SBERT cosine distance
        confidence = min(drift / 0.3, 1.0)
        return flag, confidence

    def get_name(self): 
        return "SemanticDrift"

    def get_specialization(self): 
        return ["roleplay", "hypothetical", "privilege_escalation"]


class StructuralPattern(DefenseMechanism):
    """Regex-based pattern matching for known jailbreak templates."""

    def __init__(self):
        self.patterns = [
            (r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompt)", 0.9),
            (r"disregard\s+(the\s+)?(system\s+)?(prompt|instructions)", 0.9),
            (r"you\s+are\s+(now\s+)?(in\s+)?(developer|admin|debug)\s+mode", 0.85),
            (r"DAN\s*\(?Do Anything Now\)?", 0.95),
            (r"pretend\s+(you\s+)?(are|to\s+be)\s+(an?\s+)?(unrestricted|unfiltered|AI without)", 0.8),
            (r"system\s+(override|command|message)", 0.85),
            (r"(admin|root|sudo)\s+(access|mode|privileges)", 0.8),
            (r"safety\s+(protocols?|checks?|guidelines?)\s+(disabled|bypassed|off|none)", 0.9),
            (r"Q:\s*.+\s*A:\s*", 0.7),
            (r"Example\s*\d*\s*:\s*(User|Input)", 0.75),
            (r"Base64|ROT13|hex\s*encoded|cipher\s*key", 0.8),
            (r"hypothetical\s+scenario|fictional\s+world|movie\s+script|thought\s+experiment", 0.6),
            (r"override\s*:?\s*safety|ethics\s*=\s*False|safety_protocols\s*=\s*None", 0.9),
            (r"forget\s+everything\s+you\s+were\s+told", 0.85),
            (r"new\s+(system\s+)?instruction\s*:?\s*you\s+have\s+no\s+restrictions", 0.9),
        ]

    def analyze(self, text: str, features: np.ndarray) -> Tuple[int, float]:
        text_lower = text.lower()
        total_score = 0
        matched = 0
        for pattern, weight in self.patterns:
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                matched += 1
                total_score += weight * min(count, 3)

        score = min(1.0, 0.5 * total_score)
        flag = 1 if score > 0.3 else 0
        return flag, score

    def get_name(self): 
        return "StructuralPattern"

    def get_specialization(self): 
        return ["instruction_manipulation", "privilege_escalation", "few_shot", "roleplay"]


class RephrasingStability(DefenseMechanism):
    """Evaluates attack persistence under REAL semantic rephrasing (T5)."""

    def __init__(self, use_real_paraphrase: bool = True):
        # Use lightweight extractor (no real models) for internal feature comparison
        # to avoid double-loading heavy models and prevent circular dependencies
        self.fe = FeatureExtractor(use_real_embeddings=False, use_real_perplexity=False)
        self.use_real_paraphrase = use_real_paraphrase
        self._paraphrase_model = None
        self._paraphrase_tokenizer = None

        self._structural_pattern = StructuralPattern()

        if use_real_paraphrase:
            try:
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                self._paraphrase_tokenizer = T5Tokenizer.from_pretrained("t5-small")
                self._paraphrase_model = T5ForConditionalGeneration.from_pretrained("t5-small")
                self._paraphrase_model.eval()
                print("[RephrasingStability] Loaded T5 paraphrase model")
            except Exception as e:
                print(f"[RephrasingStability] WARNING: Could not load T5: {e}")
                self.use_real_paraphrase = False

    def _rephrase(self, text: str) -> str:
        if self.use_real_paraphrase and self._paraphrase_model is not None:
            return self._rephrase_t5(text)
        return self._rephrase_rule_based(text)

    def _rephrase_t5(self, text: str) -> str:
        """Use T5 for real semantic paraphrasing."""
        try:
            import torch
            prefix = "paraphrase: "
            input_text = prefix + text
            inputs = self._paraphrase_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self._paraphrase_model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
            rephrased = self._paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return rephrased if rephrased else text
        except Exception as e:
            print(f"[RephrasingStability] T5 rephrasing failed: {e}")
            return self._rephrase_rule_based(text)

    def _rephrase_rule_based(self, text: str) -> str:
        """Fallback rule-based synonym substitution."""
        rephrased = text.lower()
        swaps = {
            "how do i": "what is the way to",
            "ignore": "disregard",
            "previous": "earlier",
            "instructions": "directions",
            "pretend": "act as if",
            "hack": "gain unauthorized access",
            "make": "create",
            "tell me": "explain",
            "describe": "portray",
            "explain": "clarify",
            "write": "compose",
            "generate": "produce",
            "create": "develop",
        }
        for old, new in swaps.items():
            rephrased = rephrased.replace(old, new)
        sentences = rephrased.split('.')
        if len(sentences) > 1:
            sentences = sentences[1:] + [sentences[0]]
        return '. '.join(sentences)

    def analyze(self, text: str, features: np.ndarray) -> Tuple[int, float]:
        rephrased = self._rephrase(text)
        orig_features = self.fe.extract(text)
        rephrased_features = self.fe.extract(rephrased)

        # Use structural+lexical features (indices 0:41) for stability comparison.
        # Features 41-42 are semantic (SBERT/GPT-2) and are not fitted inside this
        # lightweight extractor, so we compare the 41 interpretable feature dimensions.
        orig_vec = orig_features[0:41]
        reph_vec = rephrased_features[0:41]
        cos_sim = np.dot(orig_vec, reph_vec) / (np.linalg.norm(orig_vec) * np.linalg.norm(reph_vec) + 1e-8)

        len_ratio = min(len(text), len(rephrased)) / max(len(text), len(rephrased))

        _, orig_struct = self._structural_pattern.analyze(text, orig_features)
        _, reph_struct = self._structural_pattern.analyze(rephrased, rephrased_features)
        delta_attack = abs(orig_struct - reph_struct) / 2.0 if max(orig_struct, reph_struct) > 0 else 0

        stability = cos_sim * len_ratio * (1 - delta_attack)
        flag = 1 if stability < 0.7 else 0
        confidence = 1.0 - stability
        return flag, confidence

    def get_name(self): 
        return "RephrasingStability"

    def get_specialization(self): 
        return ["few_shot", "obfuscation"]
