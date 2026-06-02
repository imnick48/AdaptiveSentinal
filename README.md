# Adaptive Sentinel

**Dynamic Ensemble Defense Framework for LLM Jailbreak Detection**

## Architecture

```
Input Prompt
    ↓
Feature Extractor (43 dims)
    ├─ Statistical (12): length, complexity, composition
    ├─ Structural (19): patterns, formatting, flags
    ├─ Lexical (10): keywords, entropy, repetition
    └─ Semantic (2): SBERT drift + GPT-2 perplexity
    ↓
Defense Router
    ├─ Heuristic: keyword-based if-statements (Algorithm 1)
    └─ Learned: Random Forest meta-classifier (optional)
    ↓
Selected Defenses (2-3 of 4)
    ├─ PerplexityFilter (GPT-2 perplexity)
    ├─ SemanticDrift (SBERT embedding drift)
    ├─ StructuralPattern (regex templates)
    └─ RephrasingStability (T5 paraphrase)
    ↓
Weighted Ensemble
    ├─ Heuristic: 0.7*s_i + 0.3*c_i weighted voting
    └─ Learned: Gradient Boosting (optional)
    ↓
Jailbreak / Benign (thresholded)
```

## Project Structure

```
AdaptiveSentinel/
├── adaptive_sentinel/              # Main package
│   ├── __init__.py
│   ├── feature_extractor.py        
│   ├── defenses.py                 
│   ├── router.py                 
│   ├── ensemble.py                
│   ├── sentinel.py                
│   ├── dataset.py               
│   └── evaluate.py              
├── data/
│   └── realworld_jailbreak_dataset.json   
├── tests/
│   └── test_sentinel.py            
├── config.json                    
├── requirements.txt                
├── run.sh / run.bat                
└── README.md                      
```

## Dataset

**96 real prompts** from published research:

| Source | Paper | Venue |
|--------|-------|-------|
| JailbreakBench | Chao et al. | NeurIPS 2024 |
| PAIR | Chao et al. | arXiv 2023 |
| GCG | Zou et al. | arXiv 2023 |
| ArtPrompt | Jiang et al. | ACM CCS 2024 |
| MasterKey | Deng et al. | NDSS 2024 |
| Wei et al. | "Jailbroken" | NeurIPS 2023 |
| Anthropic | HH-RLHF | Anthropic 2022 |

- **48 jailbreaks** across 6 categories
- **48 benign** prompts from standard QA/programming/science
- Consistent 80/20 train/test split

## Installation

```bash
pip install -r requirements.txt
```

**Note:** Real models (SBERT, GPT-2, T5) download automatically on first run (~500MB total).

## Usage

### Basic evaluation (heuristic routing)
```bash
python main.py --mode adaptive --threshold 0.4 --output results/
```

### With learned components
```bash
python main.py --mode adaptive --learned-router --gradient-boosting --bootstrap --output results/
```

### Fallback mode (no real models)
```bash
python main.py --no-real-models --output results/
```

### As a library
```python
from adaptive_sentinel import AdaptiveSentinel

sentinel = AdaptiveSentinel(
    mode="adaptive",
    threshold=0.4,
    use_real_embeddings=True,    # SBERT
    use_real_perplexity=True,     # GPT-2
    use_real_paraphrase=True,     # T5
    use_learned_router=True,      # Random Forest
    use_gradient_boosting=True    # GB ensemble
)

# Fit on training data
sentinel.fit(X_train, y_train, categories)

# Predict
pred, score, defenses = sentinel.predict("Your prompt here")
# pred: 0=benign, 1=jailbreak
# score: confidence 0.0-1.0
# defenses: list of activated defenses
```

## Evaluation Metrics

The implementation reports:
- Accuracy, Precision, Recall, F1, AUC-ROC, FPR
- **Bootstrap confidence intervals** (1000 iterations)
- Per-category recall
- Confusion matrices
- ROC curves
- **Sample sizes** for all claims

## Citation

If you use this implementation, please cite the original paper and note that this is a corrected implementation:

```bibtex
@article{adaptive_sentinel_2026,
  title={Adaptive Sentinel: A Dynamic Ensemble Defense Framework for LLM Jailbreak Detection},
  author={Sagnick Das},
  year={2026}
}
```

## License

MIT License - Released as open-source for community contributions and real-world deployment evaluations.
