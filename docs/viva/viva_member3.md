# Viva notes - Member 3

Everyone must be able to explain the whole PathSnap pipeline; these are the topics I lead on: Random Forest, route metrics, road-graph construction, HMM emissions/transitions, Viterbi decoding, decoder tuning and classifier-only versus HMM evaluation.

## My 60-second summary of my part

I implemented the Random Forest candidate classifier and the route-sequence evaluation/decoding components. Random Forest scores each candidate road arc for a GPS observation using features such as distance, heading and neighbouring-observation consistency. Its student-test Top-1 accuracy was 0.8557 and Macro F1 was 0.9187. I then implemented an HMM/Viterbi decoder that treats candidate arcs as hidden states, uses classifier scores as emissions, and uses directed road-network distance versus GPS displacement for transitions. On student_test, HMM improved Random Forest Top-1 from 0.8781 to 0.9506 and route continuity from 0.9278 to 0.9783. However, it did not eliminate the 1.34% unmatched fraction, and the RF HMM result still averaged 1.8 decoder breaks per trip. The results show an improvement, not a perfect continuous route.

## Likely viva questions and answers

1. **Q: Why use Random Forest for map matching?**  
   **A:** It learns non-linear interactions among candidate features such as distance, heading difference and neighbouring-point consistency. Averaging many trees reduces the variance of a single tree.

2. **Q: What does one training row represent?**  
   **A:** One pair of a GPS observation and one candidate directed road arc. The binary target indicates whether that candidate is the correct arc.

3. **Q: What is Top-1 accuracy in this project?**  
   **A:** For each GPS observation, rank its candidate arcs by model score and check whether the highest-scoring candidate matches the labelled arc. It is observation-level, unlike candidate-row precision, recall and Macro F1.

4. **Q: What were the Random Forest results?**  
   **A:** On student_test, Top-1 was 0.8557, Macro F1 0.9187, precision 0.8852, recall 0.8217 and log loss 0.0704. Five-fold CV Macro F1 was 0.9129 ± 0.0226. Source: `results/experiments/rforest__default__20261118-131637.json`.

5. **Q: Why does the model need an HMM after classification?**  
   **A:** The classifier scores each observation's candidates independently. HMM adds sequence context and road-network connectivity to select a globally more plausible sequence.

6. **Q: What is the hidden state?**  
   **A:** A candidate directed road arc for the current GPS observation.

7. **Q: What is an emission probability?**  
   **A:** It represents how compatible a candidate state is with the current observation. We normalise the classifier scores for that observation and use their log probabilities.

8. **Q: What is a transition probability?**  
   **A:** It measures how plausible it is to move from a candidate arc at the previous observation to another candidate arc at the current observation. Our transition score penalises the difference between road-network distance and straight-line GPS displacement, and penalises reverse-twin transitions.

9. **Q: Why use Viterbi?**  
   **A:** Viterbi uses dynamic programming to find the most likely sequence of hidden states. It stores the best cumulative score and a back-pointer for each candidate, then backtracks to recover the path.

10. **Q: What is a BrokenChain?**  
    **A:** It occurs when no finite transition reaches any candidate at the next observation. The decoder closes the current segment and starts a new one rather than pretending the route is connected.

11. **Q: How did you tune the HMM?**  
    **A:** I searched combinations of `beta_m` and `uturn_penalty` using training out-of-fold scores only. The sweep used 20 training trips. The best recorded combination was beta 5 m and U-turn penalty 5, with mean sweep Top-1 0.9318. Student_test was reserved for final comparison.

12. **Q: Did HMM improve Random Forest?**  
    **A:** Yes, in aggregate student_test results. Top-1 rose from 0.8781 to 0.9506, and route continuity rose from 0.9278 to 0.9783.

13. **Q: Did HMM solve all route problems?**  
    **A:** No. RF and Logistic Regression still had an unmatched fraction of 0.013365, around 1.34%, because HMM cannot select a road arc absent from the candidate inputs. Their HMM runs also averaged 1.8 decoder breaks per trip, compared with 0.2 for Decision Tree, KNN and XGBoost.

14. **Q: Why is classifier-only `n_breaks = 0` in the summary?**  
    **A:** Classifier-only is independent per-observation argmax and does not run the sequence decoder. Zero is the summary's non-applicable placeholder; it is not proof of a connected route.

15. **Q: What is the difference between route continuity and Top-1?**  
    **A:** Top-1 checks whether the selected candidate matches the label at each observation. Route continuity checks whether consecutive predicted arcs connect on the directed road graph. A model can have good Top-1 but still produce a discontinuous route.

16. **Q: What is the main limitation of HMM?**  
    **A:** It depends on candidate coverage, graph connectivity and transition parameters. Missing candidates, large GPS gaps or impossible transitions can leave observations unmatched or split the route into multiple segments.

## Numbers I must know

- Random Forest student_test: Top-1 **0.8557**, Macro F1 **0.9187**, precision **0.8852**, recall **0.8217**, log loss **0.0704**.
- Random Forest 5-fold CV: Top-1 **0.8501 ± 0.0414**, Macro F1 **0.9129 ± 0.0226**, log loss **0.0824 ± 0.0181**.
- Random Forest training rows: **558,237** across **78 trips**; student_test: **18,631 labelled observations** and **195,635 candidate rows**.
- Random Forest train time: **27.69 s**; inference: **2.018 ms per 1,000 candidate rows**.
- HMM sweep: **20 training trips**, beta **5 m**, U-turn penalty **5**, mean Top-1 **0.9318**, mean route continuity **0.8634**.
- RF classifier-only → HMM: Top-1 **0.8781 → 0.9506**; route continuity **0.9278 → 0.9783**.
- Logistic Regression classifier-only → HMM: Top-1 **0.8624 → 0.9194**; route continuity **0.9256 → 0.9756**.
- RF and LR HMM unmatched fraction: **0.013365** (~**1.34%**); mean breaks **1.8 per trip**.
- Decision Tree, KNN and XGBoost HMM mean breaks: **0.2 per trip**.

## Files to know

- `src/models/rforest.py` — Random Forest model wrapper.
- `configs/models/rforest.yaml` — Random Forest hyperparameters.
- `src/eval/route_metrics.py` — route continuity and route-level metrics.
- `src/hmm/graph.py` — directed road graph and bounded Dijkstra.
- `src/hmm/viterbi.py` — emission, transition and Viterbi recurrence.
- `src/hmm/decode.py` — trip segmentation and route decoding.
- `src/hmm/tune_decoder.py` — training-OOF-only parameter sweep.
- `src/eval/compare_hmm.py` — classifier-only versus HMM evaluation.
- `results/experiments/rforest__default__20261118-131637.json` — Random Forest metrics.
- `results/hmm/clf_vs_hmm.csv` — aggregate student-test comparison.
- `results/hmm/per_trip.csv` — per-trip HMM results.
- `results/hmm/decoder_sweep.csv` — training OOF parameter sweep.
