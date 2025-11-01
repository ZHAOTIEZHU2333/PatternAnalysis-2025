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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Directory with feats.npy / labels.npy / edge_index.npy")
    ap.add_argument("--ckpt", default="recognition/gnn-pagepage-s4806443/outputs/ckpts/best.pt",
                    help="Checkpoint to load (.pt)")
    ap.add_argument("--fig_dir", default="recognition/gnn-pagepage-s4806443/figs",
                    help="Where to save figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden dim (must match training)")
    ap.add_argument("--dropout", type=float, default=0.5, help="Dropout (must match training)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_tsne", action="store_true", help="Disable t-SNE plotting")
    ap.add_argument("--sample_per_class", type=int, default=400, help="t-SNE per-class sample cap")
    args = ap.parse_args()

    device = torch.device(args.device)
    figdir = Path(args.fig_dir); figdir.mkdir(parents=True, exist_ok=True)

    # 1) Load tensors & masks
    x, y, edge_index, train_m, val_m, test_m = load_pagepage(
        data_dir=args.data_dir, device=device, ensure_undirected=False
    )
    N, Fdim = x.size(0), x.size(1)
    C = int(y.max().item() + 1)
    print(f"[Data] N={N} F={Fdim} C={C} test={int(test_m.sum())}")

    # 2) Build normalized adjacency (undirected + self-loops)
    A_norm = build_gcn_norm(
        edge_index=edge_index, num_nodes=N, add_loops=True, make_undirected=True, device=device
    )

    # 3) Build model and load checkpoint
    model = GCN(in_dim=Fdim, hidden_dim=args.hidden, out_dim=C, dropout=args.dropout).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print(f"[Load] checkpoint <- {args.ckpt}")

    # 4) Evaluate on test set
    te_loss, te_acc, te_pred = evaluate(model, x, A_norm, y, test_m)
    y_true = y[test_m].detach().cpu().numpy()
    y_pred = te_pred[test_m].detach().cpu().numpy()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"[Test] acc={te_acc:.4f} loss={te_loss:.4f} macro-F1={macro_f1:.4f}")

    # 5) Save confusion matrix
    plot_confusion(y_true, y_pred, figdir / "confusion_matrix.png")

    # 6) Optional t-SNE
    if not args.no_tsne:
        tsne_plot(model, x, A_norm, y, figdir / "tsne.png",
                  sample_per_class=args.sample_per_class, seed=args.seed)
        print(f"[Save] figs -> {figdir/'confusion_matrix.png'}, {figdir/'tsne.png'}")
    else:
        print(f"[Save] figs -> {figdir/'confusion_matrix.png'} (t-SNE disabled)")


if __name__ == "__main__":
    main()