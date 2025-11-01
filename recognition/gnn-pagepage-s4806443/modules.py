"""
Model components for the Facebook Page-Page node classification task.

This file is model-only: layers, networks, and graph-normalization helpers.
Training/IO logic should live in train.py / dataset.py.
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Graph helpers (sparse)
# -----------------------------
def to_undirected_coalesced(edge_index: torch.LongTensor, num_nodes: int) -> torch.sparse.FloatTensor:
    """
    Make edges undirected and coalesce duplicates. No self-loops are added here.
    Args:
        edge_index: LongTensor [2, E]
        num_nodes : number of nodes N
    Returns:
        Sparse COO adjacency A (N x N) with values=1.
    """
    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()
    ei = edge_index
    ei_rev = torch.stack((ei[1], ei[0]), dim=0)
    ei = torch.cat([ei, ei_rev], dim=1)
    # remove self loops (add later if needed)
    mask = ei[0] != ei[1]
    ei = ei[:, mask]
    vals = torch.ones(ei.size(1), dtype=torch.float32, device=ei.device)
    A = torch.sparse_coo_tensor(ei, vals, size=(num_nodes, num_nodes))
    return A.coalesce()


def add_self_loops(A: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
    """Add identity to sparse adjacency."""
    n = A.size(0)
    idx = torch.arange(n, device=A.device)
    ei2 = torch.stack([idx, idx], dim=0)
    v2 = torch.ones(n, dtype=torch.float32, device=A.device)
    ei = torch.cat([A.indices(), ei2], dim=1)
    vv = torch.cat([A.values(), v2], dim=0)
    return torch.sparse_coo_tensor(ei, vv, size=A.size()).coalesce()


def build_gcn_norm(
    edge_index: torch.LongTensor,
    num_nodes: int,
    *,
    add_loops: bool = True,
    make_undirected: bool = True,
    device: Optional[torch.device] = None,
) -> torch.sparse.FloatTensor:
    """
    Build Ĥ = D^{-1/2} Â D^{-1/2} where Â is (possibly) undirected + self-looped adjacency.
    Returns a sparse COO tensor on `device`.
    """
    device = device or edge_index.device
    # start from directed COO
    ei = edge_index.to(device)
    if make_undirected:
        A = to_undirected_coalesced(ei, num_nodes)
    else:
        vals = torch.ones(ei.size(1), dtype=torch.float32, device=device)
        A = torch.sparse_coo_tensor(ei, vals, size=(num_nodes, num_nodes)).coalesce()

    if add_loops:
        A = add_self_loops(A)

    deg = torch.sparse.sum(A, dim=1).to_dense()  # [N]
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    idx = A.indices()
    val = A.values() * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val, size=A.size()).coalesce()
