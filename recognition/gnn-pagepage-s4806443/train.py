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

    # ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory with feats.npy/labels.npy/edge_index.npy")
    ap.add_argument("--out_dir",  default="recognition/gnn-pagepage-s4806443/outputs", help="Where to save ckpts/logs")
    ap.add_argument("--fig_dir",  default="recognition/gnn-pagepage-s4806443/figs",    help="Where to save curves")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",     type=int, default=42)

    # Model/optim
    ap.add_argument("--hidden",   type=int, default=128)
    ap.add_argument("--dropout",  type=float, default=0.5)
    ap.add_argument("--lr",       type=float, default=1e-2)
    ap.add_argument("--wd",       type=float, default=5e-4)

    # Training
    ap.add_argument("--epochs",   type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Workspace
    outdir  = Path(args.out_dir)
    figdir  = Path(args.fig_dir)
    ckptdir = outdir / "ckpts"
    for d in (figdir, ckptdir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Load data tensors & masks
    x, y, edge_index, train_m, val_m, test_m = load_pagepage(
        data_dir=args.data_dir,
        device=device,
        ensure_undirected=False  # 我们在下一步用 build_gcn_norm 统一处理无向+自环
    )
    num_nodes = x.size(0)
    num_feats = x.size(1)
    num_classes = int(y.max().item() + 1)
    print(f"[Data] N={num_nodes} F={num_feats} C={num_classes}  train/val/test="
          f"{int(train_m.sum())}/{int(val_m.sum())}/{int(test_m.sum())}")

    # 2) Build normalized adjacency Ĥ
    A_norm = build_gcn_norm(
        edge_index=edge_index,
        num_nodes=num_nodes,
        add_loops=True,
        make_undirected=True,
        device=device
    )

    # 3) Model & Optimizer
    model = GCN(in_dim=num_feats, hidden_dim=args.hidden, out_dim=num_classes, dropout=args.dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # 4) Train loop with early stopping (on val acc)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = -1.0
    best_state = None
    wait = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x, A_norm)
        loss = F.cross_entropy(logits[train_m], y[train_m])
        loss.backward()
        optim.step()

        tr_loss, tr_acc = evaluate(model, x, A_norm, y, train_m)
        va_loss, va_acc = evaluate(model, x, A_norm, y, val_m)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val
        if improved:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 10 == 0 or ep == 1 or improved:
            print(f"[{ep:04d}] train {tr_acc:.4f}/{tr_loss:.4f} | val {va_acc:.4f}/{va_loss:.4f} | wait {wait}/{args.patience}")

        if wait >= args.patience:
            print(f"Early stopped at epoch {ep}")
            break

    # 5) Restore best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), ckptdir / "best.pt")
        print(f"[Save] best checkpoint -> {ckptdir / 'best.pt'}")

    te_loss, te_acc = evaluate(model, x, A_norm, y, test_m)
    print(f"[Test] acc={te_acc:.4f} loss={te_loss:.4f}")

    # 6) Save curves
    plot_curves(history, figdir)
    print(f"[Save] curves -> {figdir/'loss_curve.png'}, {figdir/'acc_curve.png'}")


if __name__ == "__main__":
    main()