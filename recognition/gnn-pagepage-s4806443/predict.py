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


@torch.no_grad()
def evaluate(model, x, A_norm, y, mask):
    model.eval()
    logits = model(x, A_norm)
    pred = logits.argmax(dim=1)
    loss = F.cross_entropy(logits[mask], y[mask]).item()
    acc = (pred[mask] == y[mask]).float().mean().item()
    return loss, acc, pred


def plot_confusion(y_true_np, y_pred_np, save_path: Path, title="Confusion Matrix"):
    from itertools import product
    cm = confusion_matrix(y_true_np, y_pred_np)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # annotate
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def tsne_plot(model, x, A_norm, y, save_path: Path, sample_per_class=400, seed=42):
    model.eval()
    logits, h = model(x, A_norm, return_hidden=True)  
    y_np = y.detach().cpu().numpy()
    h_np = h.detach().cpu().numpy()

    
    rng = np.random.default_rng(seed)
    idxs = []
    for c in np.unique(y_np):
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        idxs.append(idx_c[:sample_per_class])
    sel = np.concatenate(idxs) if len(idxs) else np.arange(len(y_np))

    Z = TSNE(n_components=2, init="pca", learning_rate="auto",
             perplexity=30, random_state=seed).fit_transform(h_np[sel])

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=y_np[sel], s=5, alpha=0.85, cmap="tab20")
    plt.title("t-SNE of Node Embeddings (colored by true label)")
    plt.xticks([]); plt.yticks([])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()