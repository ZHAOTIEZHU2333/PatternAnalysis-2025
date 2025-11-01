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