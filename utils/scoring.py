from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


@torch.no_grad()
def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T


@torch.no_grad()
def mutual_nearest_neighbor_scores(query_feats: torch.Tensor | None, ref_feats: torch.Tensor | None) -> torch.Tensor:
    if query_feats is None or ref_feats is None:
        return torch.empty(0)
    if query_feats.numel() == 0 or ref_feats.numel() == 0:
        return torch.empty(0)

    sim = cosine_similarity(query_feats, ref_feats)
    q_to_r = sim.argmax(dim=1)
    r_to_q = sim.argmax(dim=0)

    scores = []
    for qi, rj in enumerate(q_to_r.tolist()):
        if r_to_q[rj].item() == qi:
            scores.append(sim[qi, rj].item())
    if not scores:
        return torch.empty(0)
    return torch.tensor(scores, dtype=torch.float32)


@torch.no_grad()
def aggregate(scores: torch.Tensor, mode: Literal["mean", "max"] = "max") -> float:
    if scores.numel() == 0:
        return 0.0
    if mode == "mean":
        return float(scores.mean().item())
    return float(scores.max().item())


@torch.no_grad()
def object_aware_score(
    global_score: float,
    q_patches: torch.Tensor | None,
    r_patches: torch.Tensor | None,
    q_objects: torch.Tensor | None,
    r_objects: torch.Tensor | None,
    patch_mode: Literal["mean", "max"] = "max",
    object_mode: Literal["mean", "max"] = "mean",
) -> dict:
    patch_scores = mutual_nearest_neighbor_scores(q_patches, r_patches)
    object_scores = mutual_nearest_neighbor_scores(q_objects, r_objects)
    s_patch = aggregate(patch_scores, patch_mode)
    s_object = aggregate(object_scores, object_mode)
    total = float(global_score) + s_patch + s_object
    return {
        "global": float(global_score),
        "patch": s_patch,
        "object": s_object,
        "total": total,
        "num_patch_matches": int(patch_scores.numel()),
        "num_object_matches": int(object_scores.numel()),
    }
