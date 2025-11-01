"""
Training script for Facebook Page-Page node classification (GCN baseline).
- Loads feats/labels/edge_index and stratified masks from dataset.py
- Builds normalized adjacency (Ĥ = D^{-1/2} Â D^{-1/2})
- Trains 2-layer GCN with early stopping
- Saves: best checkpoint (outputs/ckpts/best.pt) and curves (figs/*.png)
"""

from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import load_pagepage
from modules import GCN, build_gcn_norm



# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module,
             x: torch.Tensor,
             A_norm: torch.sparse.FloatTensor,
             y: torch.Tensor,
             mask: torch.Tensor) -> Tuple[float, float]:
    """Return (loss, accuracy) on the given mask."""
    model.eval()
    logits = model(x, A_norm)
    loss = F.cross_entropy(logits[mask], y[mask]).item()
    pred = logits.argmax(dim=1)
    acc = (pred[mask] == y[mask]).float().mean().item()
    return loss, acc


def plot_curves(history: Dict[str, List[float]], figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"],   label="val")
    plt.title("Loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.tight_layout()
    (figdir / "loss_curve.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figdir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"],   label="val")
    plt.title("Accuracy")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "acc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()