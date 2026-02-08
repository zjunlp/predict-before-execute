# Evaluation metrics for ranking/selection tasks.
from typing import List, Tuple
import math

def accuracy_n2(predicted_idx: int, gt_idx: int) -> int:
    return 1 if int(predicted_idx) == int(gt_idx) else 0

def spearman_rho_from_rankings(predicted_ranking: List[int], ground_ranking: List[int]) -> float:
    """
    predicted_ranking and ground_ranking are lists of indices ordered from best to worst.
    They must contain the same elements (possibly same length).
    Compute Spearman's rho: 1 - 6 * sum(d^2) / (n*(n^2-1))
    """
    if not predicted_ranking or not ground_ranking:
        return 0.0
    n = len(ground_ranking)
    # build rank maps: index -> rank (1..n)
    rank_pred = {idx: rank + 1 for rank, idx in enumerate(predicted_ranking)}
    rank_gt = {idx: rank + 1 for rank, idx in enumerate(ground_ranking)}
    # ensure all ground indices present in predicted; if missing, assign worst rank (n)
    dsq = 0.0
    for idx in ground_ranking:
        rp = rank_pred.get(idx, n)
        rg = rank_gt.get(idx, n)
        d = rp - rg
        dsq += d * d
    denom = n * (n * n - 1)
    if denom == 0:
        return 0.0
    rho = 1.0 - (6.0 * dsq) / denom
    return rho

def topk_accuracy(predicted_ranking: List[int], ground_best_index: int, k: int) -> int:
    """
    Returns 1 if ground_best_index is within predicted top-k, else 0.
    """
    topk = predicted_ranking[:k]
    return 1 if int(ground_best_index) in topk else 0

def precision_at_k(predicted_ranking: List[int], ground_ranking: List[int], k: int) -> float:
    """
    Exact prefix-match accuracy at k:
    Returns 1.0 iff predicted_ranking[:k] equals ground_ranking[:k] element-wise and in order; else 0.0.
    Only meaningful for 1 <= k <= len(ground_ranking).
    """
    if not predicted_ranking or not ground_ranking:
        return 0.0
    k = max(1, min(k, len(ground_ranking)))
    return 1.0 if predicted_ranking[:k] == ground_ranking[:k] else 0.0