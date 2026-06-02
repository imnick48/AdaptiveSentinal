"""
Dataset Loader - Handles consistent splits
"""

import json
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def load_dataset(filepath: str = None) -> Tuple[List[str], List[int], List[str]]:
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'realworld_jailbreak_dataset.json')
        filepath = os.path.abspath(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    texts = []
    labels = []
    categories = []

    for item in data.get("jailbreaks", []):
        texts.append(item["text"])
        labels.append(1)
        categories.append(item.get("category", "unknown"))

    for item in data.get("benign", []):
        texts.append(item["text"])
        labels.append(0)
        categories.append("benign")

    return texts, labels, categories


def split_dataset(texts: List[str], labels: List[int], categories: List[str],
                  test_size: float = 20/96, random_state: int = 42):
    return train_test_split(
        texts, labels, categories,
        test_size=test_size, random_state=random_state, stratify=categories
    )
