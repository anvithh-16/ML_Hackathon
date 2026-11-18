"""Viterbi on a hand-built lattice where greedy is wrong. Owner: Member 3."""
import numpy as np
import pytest

from src.hmm.viterbi import BrokenChain, emission_logprob, viterbi, viterbi_step

L = np.log


def test_viterbi_beats_greedy():
    e = [L([0.6, 0.4]), L([0.55, 0.45]), L([0.5, 0.5])]
    # state 0 can only stay 0 cheaply; greedy picks 0,0,? but the chain via 1 is far more likely
    t = [L([[0.01, 0.99], [0.01, 0.99]]), L([[0.5, 0.5], [0.01, 0.99]])]
    path, lp = viterbi(e, t)
    assert path == [0, 1, 1]
    assert abs(lp - (L(0.6) + L(0.99) + L(0.45) + L(0.99) + L(0.5))) < 1e-9


def test_broken_chain():
    with pytest.raises(BrokenChain):
        viterbi_step(np.zeros(2), np.full((2, 2), -np.inf), np.zeros(2))


def test_emission_normalised():
    assert abs(np.exp(emission_logprob(np.array([1.0, 3.0]), 0)).sum() - 1) < 1e-9
