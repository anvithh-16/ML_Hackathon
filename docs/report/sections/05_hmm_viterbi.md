# 5. HMM and Viterbi route reconstruction (owner: Member 3)

Implementation: `src/hmm/graph.py`, `src/hmm/viterbi.py`, `src/hmm/decode.py`  
Configuration: `configs/hmm.yaml`  
Comparison results: `results/hmm/clf_vs_hmm.csv` and `results/hmm/per_trip.csv`  
Training-only decoder sweep: `results/hmm/decoder_sweep.csv`

## 5.1 Model formulation

**1. Hidden states.** At each GPS observation, the hidden state is one of the candidate directed road arcs generated for that observation. The decoder chooses a sequence of candidate arcs rather than making each observation's decision independently.

**2. Emission probabilities.** For classifier-based decoding, candidate scores for an observation are normalised to sum to one and converted to log probabilities:

`log e_i(b) = log(score_b / sum(scores for observation i) + epsilon)`

A distance-only baseline is also implemented using Gaussian distance scores with `sigma_m = 5.0`.

**3. Transition probabilities.** A transition between candidate arcs is evaluated using road-network distance between their projected positions and the straight-line GPS displacement. The log transition score penalises the absolute difference between these distances, scaled by `beta_m`, and applies an additional penalty when moving to the reverse twin arc:

`log T(a -> b) = -abs(network_distance(a,b) - GPS_straight_distance) / beta_m - uturn_penalty * I(b is reverse(a))`

Network distance is computed over the directed road graph using bounded, memoised Dijkstra searches. The search limit is `dijkstra_limit_factor * straight_distance + dijkstra_limit_extra_m`.

**4. Viterbi decoding.** Dynamic programming keeps the best cumulative log score for each candidate state and stores back-pointers. Backtracking recovers the most likely arc sequence for a segment. A new segment starts at configured time gaps, large GPS jumps, observations with no candidates, or when no finite transition reaches the next observation (`BrokenChain`). Missing-candidate observations remain unmatched (`pred_arc_id = -1`); HMM cannot invent a candidate that candidate generation did not provide.

## 5.2 Parameter selection and evaluation protocol

Decoder parameters were explored using training out-of-fold scores only, not student_test. The sweep evaluated 15 combinations of `beta_m` and `uturn_penalty` on 20 training trips. The best recorded combination was `beta_m = 5` and `uturn_penalty = 5`, with mean HMM Top-1 of 0.9318 and mean route continuity of 0.8634 (`results/hmm/decoder_sweep.csv`). These values match `configs/hmm.yaml`.

The final classifier-only versus classifier+HMM comparison was run on the 10-trip `student_test` partition and is summarised in `results/hmm/clf_vs_hmm.csv`. Per-trip variation is available in `results/hmm/per_trip.csv`.

## 5.3 Student-test comparison

The following are aggregate values from `results/hmm/clf_vs_hmm.csv`. Top-1 and route continuity are proportions; `n_breaks` is the mean number of decoder breaks per trip; `frac_unmatched` is the fraction of observations without a candidate prediction.

| Score source | Method | Top-1 | Route continuity | Mean breaks | Unmatched fraction |
|---|---|---:|---:|---:|---:|
| Logistic Regression | classifier-only | 0.8624 | 0.9256 | 0.0* | 0.01337 |
| Logistic Regression | classifier + HMM | 0.9194 | 0.9756 | 1.8 | 0.01337 |
| Random Forest | classifier-only | 0.8781 | 0.9278 | 0.0* | 0.01337 |
| Random Forest | classifier + HMM | 0.9506 | 0.9783 | 1.8 | 0.01337 |
| Decision Tree | classifier-only | 0.8703 | 0.9230 | 0.0* | 0.00000 |
| Decision Tree | classifier + HMM | 0.9479 | 0.9785 | 0.2 | 0.00000 |
| KNN | classifier-only | 0.8510 | 0.9303 | 0.0* | 0.00000 |
| KNN | classifier + HMM | 0.9098 | 0.9773 | 0.2 | 0.00000 |
| XGBoost | classifier-only | 0.8798 | 0.9354 | 0.0* | 0.00000 |
| XGBoost | classifier + HMM | 0.9423 | 0.9788 | 0.2 | 0.00000 |

`*` Classifier-only independently selects the top candidate per observation and does not run sequence decoding. Its zero `n_breaks` is a placeholder/not-applicable value, not evidence that its predictions form a connected path. Do not compare this zero directly with the HMM break count.

## 5.4 Where HMM helped

- HMM increased observation-level Top-1 for every classifier in the aggregate student-test comparison.
- For Random Forest, Top-1 increased from 0.8781 to 0.9506 (+7.26 percentage points), and route continuity increased from 0.9278 to 0.9783.
- For Logistic Regression, Top-1 increased from 0.8624 to 0.9194 (+5.70 percentage points), and route continuity increased from 0.9256 to 0.9756.
- HMM uses road-network connectivity and temporal context to correct locally plausible but globally inconsistent candidate choices.

## 5.5 Where HMM did not fully help / limitations

1. **Unmatched observations remain unmatched.** Random Forest and Logistic Regression both have `frac_unmatched = 0.013365` (about 1.34%) before and after HMM. The decoder cannot recover the true arc if candidate generation did not supply it.
2. **More decoder breaks for RF and Logistic Regression.** The aggregate HMM results report 1.8 breaks per trip for both Random Forest and Logistic Regression, compared with 0.2 for Decision Tree, KNN and XGBoost. HMM improves average Top-1 and continuity, but the RF/LR score inputs still lead to more cases where the decoder must split the sequence into separate segments.
3. **Difficult trips remain difficult.** For trip `00000056`, the unmatched fraction is 0.13365 for both RF and Logistic Regression; HMM reports 16 breaks for each. Their Top-1 and route precision improve, but the trip is not recovered as one uninterrupted chain.
4. **The decoder depends on graph quality and parameters.** A missing/disconnected road-network path, large GPS gap, or restrictive transition search can trigger a segment break.

**Conclusion.** HMM/Viterbi improved aggregate Top-1 and route continuity across the evaluated models, but did not remove unmatched observations. RF and Logistic Regression retained a higher mean break count (1.8 per trip) than the other compared models (0.2). HMM improves route reconstruction overall, but its benefit is limited by candidate coverage and model-dependent transition-chain failures.

## 5.6 Reproduction commands

```bash
# Tune only on training out-of-fold scores
python -m src.hmm.tune_decoder --scores rforest__default --n-trips 20

# Compare classifier-only and HMM on student_test
python -m src.eval.compare_hmm --models logreg__default rforest__default dtree__default knn__default xgb__default --partition student_test
```

The decoder sweep and final comparison files cited above are the source of the reported values. Do not tune decoder parameters on student_test.
