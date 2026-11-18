# 5. HMM and Viterbi route reconstruction  (owner: Member 3)

> Organiser rule: answers must connect the algorithm's behaviour to OUR experiments. Every number must come from a file
> in `results/` (cite the file). Write in your own words. Do not paste numbers you did not produce.

1. Hidden states (candidate arcs per observation).
2. Emission probabilities (normalised classifier scores; distance-only baseline).
3. Transition probabilities (network vs straight-line distance, beta, U-turn penalty, search limit).
4. Missing candidates and large GPS gaps (segments).
5. Disconnected consecutive segments (BrokenChain -> new segment; path fill-in).
6. Did HMM decoding improve the route? Table from `results/hmm/clf_vs_hmm.csv` (classifier-only vs classifier+HMM: Top-1, continuity, route precision/recall, breaks), per-trip `results/hmm/per_trip.csv`; cases where it did not help.
- Decoder parameters tuned on training out-of-fold scores: `results/hmm/decoder_sweep.csv`.
- Maps: `results/hmm/maps/*.html` (reference vs classifier-only vs HMM).
