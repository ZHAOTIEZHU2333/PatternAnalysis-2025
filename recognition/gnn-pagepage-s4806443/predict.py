"""
Inference / Evaluation for Facebook Page-Page (GCN baseline)

- Loads feats/labels/edge_index and masks via dataset.py
- Builds normalized adjacency Ĥ using modules.py
- Restores checkpoint and reports Test metrics (Accuracy, Macro-F1)
- Saves confusion matrix and (optional) t-SNE embedding plot

Outputs:
  figs/confusion_matrix.png
  figs/tsne.png               (unless --no_tsne)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.manifold import TSNE

from dataset import load_pagepage
from modules import GCN, build_gcn_norm