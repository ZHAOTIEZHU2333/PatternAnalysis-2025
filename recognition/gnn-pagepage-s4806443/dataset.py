"""
Dataset utilities for the Facebook Large Page-Page node classification task.

"""
from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import torch

__all__ = ["load_pagepage", "stratified_masks", "load_edge_index"]

def _must_exist(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

def load_edge_index(data_dir: str, device: torch.device) -> torch.LongTensor:
    """
    Load edge_index from either `edge_index.npy` or `edges.txt` (space-separated `src dst`).
    Returns LongTensor [2, E] on `device`.
    """
    npy_path = os.path.join(data_dir, "edge_index.npy")
    txt_path = os.path.join(data_dir, "edges.txt")

    if os.path.exists(npy_path):
        ei = np.load(npy_path)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"`edge_index.npy` must be [2, E], got {ei.shape}")
        return torch.from_numpy(ei.astype(np.int64, copy=False)).to(device)

    if os.path.exists(txt_path):
        edges = np.loadtxt(txt_path, dtype=np.int64)
        if edges.ndim == 1:
            edges = edges[None, :]  # single edge
        if edges.shape[1] != 2:
            raise ValueError(f"`edges.txt` must have 2 columns, got {edges.shape[1]}")
        return torch.from_numpy(edges.T.astype(np.int64, copy=False)).to(device)

    raise FileNotFoundError(f"Neither `edge_index.npy` nor `edges.txt` found under: {data_dir}")

def stratified_masks(
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build boolean train/val/test masks with per-class stratification.
    """
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1.0:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio and train_ratio + val_ratio < 1.")

    rng = np.random.default_rng(seed)
    n = y.shape[0]
    train = np.zeros(n, dtype=bool)
    val = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)

    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train = min(int(round(len(idx) * train_ratio)), len(idx))
        n_val   = min(int(round(len(idx) * val_ratio)),  max(0, len(idx) - n_train))
        train[idx[:n_train]] = True
        val[idx[n_train:n_train+n_val]] = True
        test[idx[n_train+n_val:]] = True

    assert not (train & val).any() and not (train & test).any() and not (val & test).any(), "Masks must be disjoint."
    assert (train | val | test).all(), "Every sample must belong to one of the splits."
    return train, val, test