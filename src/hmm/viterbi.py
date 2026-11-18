"""HMM building blocks and the Viterbi recurrence (pure numpy). Owner: Member 3.

Hidden state at observation i  = one of its candidate arcs.
Emission   log e_i(b) = log( score_b / sum(scores of obs i) + eps )      (any model's scores)
Transition log T(a->b) = -| network_dist(a,b) - straight_dist(i-1,i) | / beta  - uturn_penalty*[b == reverse(a)]
Viterbi    delta_i(b) = log e_i(b) + max_a [ delta_{i-1}(a) + log T(a->b) ],  back-pointer = argmax
"""
from __future__ import annotations

import numpy as np


class BrokenChain(Exception):
    """No finite path into the current observation (all transitions impossible)."""


def emission_logprob(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = np.asarray(scores, float)
    tot = s.sum()
    if tot <= 0 or not np.isfinite(tot):
        return np.full(len(s), np.log(1.0 / max(len(s), 1)))
    return np.log(s / tot + eps)


def distance_scores(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """Distance-only emission baseline (Gaussian GPS noise)."""
    return np.exp(-0.5 * (np.asarray(dist_m, float) / sigma_m) ** 2) + 1e-12


def transition_logprob(net_d: np.ndarray, gc_d: float, beta_m: float, is_uturn: np.ndarray,
                       uturn_penalty: float) -> np.ndarray:
    lt = -np.abs(np.asarray(net_d, float) - gc_d) / beta_m - uturn_penalty * np.asarray(is_uturn, float)
    lt[~np.isfinite(np.asarray(net_d, float))] = -np.inf
    return lt


def viterbi_step(delta_prev: np.ndarray, log_t: np.ndarray, log_e: np.ndarray):
    """One recurrence step. log_t has shape (n_prev, n_cur). Returns (delta, backpointers)."""
    m = delta_prev[:, None] + log_t
    bp = np.argmax(m, axis=0)
    best = m[bp, np.arange(m.shape[1])]
    delta = best + log_e
    if not np.isfinite(delta).any():
        raise BrokenChain()
    return delta, bp


def backtrack(deltas_last: np.ndarray, bps: list[np.ndarray]) -> list[int]:
    k = int(np.argmax(deltas_last))
    path = [k]
    for bp in reversed(bps):
        k = int(bp[k])
        path.append(k)
    return path[::-1]


def viterbi(log_e: list[np.ndarray], log_t: list[np.ndarray]) -> tuple[list[int], float]:
    """Full Viterbi. log_t[i] is the (n_{i}, n_{i+1}) matrix between steps i and i+1."""
    delta = np.asarray(log_e[0], float)
    bps = []
    for i in range(1, len(log_e)):
        delta, bp = viterbi_step(delta, log_t[i - 1], np.asarray(log_e[i], float))
        bps.append(bp)
    path = backtrack(delta, bps)
    return path, float(np.max(delta))
