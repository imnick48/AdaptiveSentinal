"""
- Multi-dimensional Feature Extraction Module (43 features)
- Uses real SBERT embeddings and real GPT-2 perplexity
"""

import numpy as np
import re
import math
from collections import Counter
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extracts 43-dimensional feature vectors from prompts.

    Features:
    - Statistical (12): length, complexity, composition
    - Structural (19): patterns, formatting, attack flags
    - Lexical (10): keyword counts, entropy, repetition
    - Semantic (2): REAL SBERT embedding drift + REAL GPT-2 perplexity
    """

    def __init__(self, use_real_embeddings: bool = True, use_real_perplexity: bool = True):
        self.use_real_embeddings = use_real_embeddings
        self.use_real_perplexity = use_real_perplexity

        # Load real models if requested
        self._sbert_model = None
        self._gpt2_model = None
        self._gpt2_tokenizer = None

        if use_real_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("[FeatureExtractor] Loaded SBERT model (all-MiniLM-L6-v2)")
            except Exception as e:
                print(f"[FeatureExtractor] WARNING: Could not load SBERT: {e}")
                self.use_real_embeddings = False

        if use_real_perplexity:
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                self._gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self._gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self._gpt2_model.eval()
                print("[FeatureExtractor] Loaded GPT-2 for perplexity computation")
            except Exception as e:
                print(f"[FeatureExtractor] WARNING: Could not load GPT-2: {e}")
                self.use_real_perplexity = False

        # Benign reference embeddings (computed from benign corpus using SBERT)
        self._benign_centroid = None

        # Keyword lists
        self.suspicious_keywords = [
            "ignore", "override", "bypass", "pretend", "hypothetical", "disregard",
            "suspend", "disable", "none", "unrestricted", "debug", "admin", "root",
            "sudo", "system", "command", "mode", "DAN", "jailbreak", "hack", "exploit",
            "malware", "virus", "bomb", "weapon", "poison", "fraud", "steal", "kill",
            "hurt", "torture", "forge", "fake", "scam", "phish", "botnet", "ddos",
            "brute", "force", "crack", "password", "identity", "theft", "drug",
            "meth", "cocaine", "synthesize", "counterfeit", "ransom", "stalk",
            "swat", "doxx", "spam", "hate", "speech", "illegal", "evade", "cheat"
        ]
        self.override_keywords = ["ignore", "override", "disregard", "suspend", "forget", "previous", "instructions"]
        self.safety_keywords = ["safety", "ethical", "guidelines", "policy", "restrictions", "constraints"]
        self.mode_keywords = ["developer", "admin", "debug", "root", "sudo", "maintenance"]
        self.persona_keywords = ["pretend", "act", "role", "character", "persona", "you are", "DAN"]
        self.framing_keywords = ["hypothetical", "fictional", "movie", "script", "novel", "game", "scenario", "academic", "theoretical"]

        # Structural regex patterns
        self.override_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompt|commands)",
            r"disregard\s+(the\s+)?(system\s+)?(prompt|instructions|guidelines)",
            r"(system\s+)?(override|command|message)",
            r"(admin|root|sudo)\s+(access|mode|privileges)",
            r"(developer|debug)\s+mode",
            r"safety\s+(protocols?|checks?|guidelines?)\s+(disabled|bypassed|off|none)"
        ]
        self.roleplay_patterns = [
            r"pretend\s+(you\s+)?(are|to\s+be)",
            r"act\s+(as|like)",
            r"you\s+are\s+(now\s+)?(an?\s+)?(unrestricted|unfiltered|AI|assistant|expert)",
            r"role\s*(play|as)",
            r"DAN\s*\(?Do Anything Now\)?"
        ]
        self.hypothetical_patterns = [
            r"in\s+a\s+(hypothetical|fictional|virtual|theoretical)\s+(world|scenario|setting|situation)",
            r"for\s+a\s+(movie|script|novel|story|game)",
            r"this\s+is\s+(purely|just)\s+(academic|theoretical|hypothetical)",
            r"thought\s+experiment",
            r"imagine\s+a\s+world"
        ]
        self.encoding_patterns = [
            r"base64", r"rot13", r"hex", r"encode", r"decode", r"cipher",
            r"[A-Za-z0-9+/]{20,}={0,2}",
            r"[0-9a-fA-F]{20,}",
            r"[\.\-/]{10,}"
        ]
        self.fewshot_patterns = [
            r"Q:\s*.+\s*A:\s*",
            r"Example\s*\d*\s*:",
            r"Shot\s*\d+",
            r"Ex\d+",
            r"\[Ex\d+\]",
            r"Input:\s*.+\s*Output:\s*",
            r"User:\s*'.+'\s*Bot:\s*'.+'"
        ]

    def fit_benign_reference(self, benign_texts: List[str]):
        """Compute benign reference centroid from benign corpus using SBERT."""
        if self.use_real_embeddings and self._sbert_model is not None:
            print(f"[FeatureExtractor] Encoding {len(benign_texts)} benign texts with SBERT ...", flush=True)
            embeddings = self._sbert_model.encode(benign_texts, convert_to_numpy=True,
                                                   show_progress_bar=False)
            self._benign_centroid = np.mean(embeddings, axis=0)
            print(f"[FeatureExtractor] Computed benign centroid from {len(benign_texts)} samples "
                  f"(dim={len(self._benign_centroid)})")
        else:
            # Fallback: use hash-based approximation
            self._benign_centroid = self._compute_hash_centroid(benign_texts)

    def _compute_hash_centroid(self, texts: List[str]) -> np.ndarray:
        """Fallback hash-based centroid (for when SBERT unavailable)."""
        vec_dim = 384  # Match SBERT dimension
        centroid = np.zeros(vec_dim)
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            vec = np.zeros(vec_dim)
            for w in words:
                h = hash(w) % vec_dim
                vec[h] += 1
            if np.linalg.norm(vec) > 0:
                vec = vec / np.linalg.norm(vec)
            centroid += vec
        return centroid / len(texts) if texts else centroid

    def _compute_real_perplexity(self, text: str) -> float:
        """Compute real perplexity using GPT-2."""
        if not self.use_real_perplexity or self._gpt2_model is None:
            return self._compute_entropy_proxy(text)

        try:
            import torch
            with torch.no_grad():
                inputs = self._gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                outputs = self._gpt2_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                perplexity = math.exp(loss)
                # Normalize: typical perplexity ranges 10-1000, normalize to 0-1 range
                normalized = min(perplexity / 100.0, 1.0)
                return normalized
        except Exception as e:
            print(f"[FeatureExtractor] Perplexity computation failed: {e}")
            return self._compute_entropy_proxy(text)

    def _compute_entropy_proxy(self, text: str) -> float:
        """Fallback character entropy proxy."""
        char_counts = Counter(text)
        if not text:
            return 0.0
        char_entropy = -sum((c/len(text)) * math.log2(c/len(text)) for c in char_counts.values())
        return char_entropy / 4.5

    def _compute_real_embedding_drift(self, text: str) -> float:
        """Compute real semantic drift using SBERT."""
        if not self.use_real_embeddings or self._sbert_model is None or self._benign_centroid is None:
            return self._compute_hash_drift(text)

        try:
            embedding = self._sbert_model.encode([text], convert_to_numpy=True)[0]
            # Cosine distance = 1 - cosine similarity
            dot = np.dot(embedding, self._benign_centroid)
            norm = np.linalg.norm(embedding) * np.linalg.norm(self._benign_centroid)
            cos_sim = dot / (norm + 1e-8)
            drift = 1 - cos_sim
            return float(drift)
        except Exception as e:
            print(f"[FeatureExtractor] Embedding drift computation failed: {e}")
            return self._compute_hash_drift(text)

    def _compute_hash_drift(self, text: str) -> float:
        """Fallback hash-based drift."""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words or self._benign_centroid is None:
            return 0.0
        vec_dim = len(self._benign_centroid)
        vec = np.zeros(vec_dim)
        for w in words:
            h = hash(w) % vec_dim
            vec[h] += 1
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        dot = np.dot(vec, self._benign_centroid)
        norm = np.linalg.norm(vec) * np.linalg.norm(self._benign_centroid)
        cos_sim = dot / (norm + 1e-8)
        return float(1 - cos_sim)

    def extract(self, text: str) -> np.ndarray:
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        lines = [l for l in text.split('\n') if l.strip()]
        chars = list(text)

        features = []

        # === STATISTICAL FEATURES (12) ===
        features.append(len(text))                           # 0: char count
        features.append(len(words))                          # 1: word count
        features.append(len(sentences))                      # 2: sentence count
        features.append(len(lines))                          # 3: line count
        features.append(np.mean([len(w) for w in words]) if words else 0)  # 4: avg word length
        features.append(np.mean([len(s.split()) for s in sentences]) if sentences else 0)  # 5: avg sentence length
        features.append(len(set(words)) / len(words) if words else 0)  # 6: unique word ratio
        features.append(sum(c.isdigit() for c in chars) / len(chars) if chars else 0)  # 7: digit ratio
        features.append(sum(c.isupper() for c in chars) / len(chars) if chars else 0)  # 8: uppercase ratio
        features.append(sum(not c.isalnum() and not c.isspace() for c in chars) / len(chars) if chars else 0)  # 9: special char ratio
        features.append(text.count('!'))                     # 10: exclamation count
        features.append(text.count('?'))                     # 11: question count

        # === STRUCTURAL FEATURES (19) ===
        features.append(1 if '```' in text or '`' in text else 0)  # 12: code blocks
        features.append(1 if '"' in text or "'" in text else 0)   # 13: quotes
        features.append(1 if any(c in text for c in '@#$%^&*') else 0)  # 14: special chars
        features.append(1 if sum(c.isupper() for c in chars) > len(chars)*0.3 else 0)  # 15: excessive caps
        features.append(1 if 'http' in text_lower or 'www.' in text_lower else 0)  # 16: URLs
        features.append(1 if any(c in text for c in '*_[]()') else 0)  # 17: markdown
        features.append(1 if any(re.search(p, text_lower) for p in self.override_patterns) else 0)  # 18: override commands
        features.append(1 if any(re.search(p, text_lower) for p in self.roleplay_patterns) else 0)    # 19: roleplay setup
        features.append(1 if any(re.search(p, text_lower) for p in self.hypothetical_patterns) else 0) # 20: hypothetical framing
        features.append(1 if any(re.search(p, text_lower) for p in self.encoding_patterns) else 0)  # 21: encoding markers
        features.append(1 if any(re.search(p, text) for p in self.fewshot_patterns) else 0)           # 22: few-shot patterns
        features.append(len(lines))                          # 23: line count (format)
        features.append(np.mean([len(l) for l in lines]) if lines else 0)  # 24: avg line length
        features.append(1 if re.search(r'\d+\.', text) else 0)  # 25: numbered lists
        features.append(1 if re.search(r'[-*]\s', text) else 0)  # 26: bullet points
        features.append(text.count(':'))                     # 27: colon count
        features.append(text.count('='))                     # 28: equals count
        features.append(text.count('['))                     # 29: bracket count
        features.append(text.count('('))                     # 30: paren count

        # === LEXICAL FEATURES (10) ===
        suspicious_count = sum(1 for w in words if w in self.suspicious_keywords)
        features.append(suspicious_count)                    # 31: suspicious keyword count
        features.append(suspicious_count / len(words) if words else 0)  # 32: suspicious keyword ratio
        features.append(sum(1 for w in words if w in self.override_keywords))  # 33: override keyword count
        features.append(sum(1 for w in words if w in self.safety_keywords))    # 34: safety keyword count
        features.append(sum(1 for w in words if w in self.mode_keywords))    # 35: mode keyword count
        features.append(sum(1 for w in words if w in self.persona_keywords))   # 36: persona keyword count
        features.append(sum(1 for w in words if w in self.framing_keywords))  # 37: framing keyword count
        if chars:
            char_counts = Counter(chars)
            entropy = -sum((c/len(chars)) * math.log2(c/len(chars)) for c in char_counts.values())
            features.append(entropy)                         # 38: character entropy
        else:
            features.append(0)
        word_counts = Counter(words)
        features.append(max(word_counts.values()) if word_counts else 0)  # 39: max word repetition
        features.append(sum(1 for v in word_counts.values() if v > 1) / len(words) if words else 0)  # 40: repeated word ratio

        # === SEMANTIC FEATURES (2) ===
        features.append(self._compute_real_embedding_drift(text))   # 41: REAL embedding drift
        features.append(self._compute_real_perplexity(text))         # 42: REAL perplexity

        return np.array(features, dtype=np.float32)
