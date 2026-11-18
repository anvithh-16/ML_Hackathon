# AI-use disclosure  (owner: Member 1)

Required by the PathSnap Assessment & Academic Integrity Policy.

- **Tool(s) used:** Claude Code (Anthropic, model Claude Opus 5.5) — used by Member 1.
  *(Other members: add the tools you used, one line each.)*
- **What it was used for (Member 1):**
  - setting up the environment, running the Member 1 pipeline on the real data (validation, folds, labels,
    candidates, features, radius sweep, EDA figures/maps, Logistic Regression runs) and creating the git branches and pull requests;
  - reviewing the starter code against its outputs, which found and fixed one inconsistency (the anomaly decision for
    trip 00000002 did not match `exclude_trips_train`, PR #11);
  - first drafts of report sections 01, 02, 03 (Logistic Regression) and 06, using only numbers from `results/`.
- **Tool(s) used (Member 2):** Claude Code (Anthropic; Claude Opus 5.5 and Claude Sonnet 5.5 in different parts of the session) — used by Member 2.
- **What it was used for (Member 2):**
  - setting up the environment and running the evaluation framework, the KNN and Decision Tree runs, the model
    comparison, the decoder tuning and the classifier-vs-HMM experiment on the real data, and creating the git branches and
    commits (committed under Member 2's own name; no AI co-author line);
  - installing and running, at Member 2's request, the starter code of the other members' branches `feat/hmm`, `feat/api`,
    `feat/tuning-export` and `feat/hmm-experiments`, and running the XGBoost default and tuned runs, then pushing them for
    review; the owners of those parts still have to review and be able to explain them;
  - the overfitting script `src/eval/train_vs_val.py`, two small fixes (relative file paths in the comparison table and in the
    export manifest, a plot legend) and the UI screenshots, which were taken with a scripted browser;
  - first drafts of report sections 03 (KNN), 03 (Decision Tree) and 07b and of `docs/viva/viva_member2.md`, using only
    numbers from files in `results/`.
- **Problems found while checking AI output (Member 2):** the AI first misread the baseline row of the comparison table
  (it reported baseline Top-1 as 0.80, the correct value is 0.54; 0.80 is its route continuity), committed some API files to
  wrong paths (fixed in a follow-up commit), and wrote two wrong ranking statements in the Decision Tree draft (rank and
  speed), found by checking the draft against `results/model_comparison.csv`. All were corrected.
- **What the team did itself:** deciding the plan and work split, choosing the model, reviewing every map and figure,
  deciding on flagged trips, rewriting the report text in our own words, and reviewing and merging every PR.
- **How we verified AI-produced code and text:** unit tests (`tests/`), checks on real data against the expected
  counts (80 / 208 417 train observations, 10 / 19 160 student_test), a nearest-candidate baseline to check metrics,
  every report number traced back to a file in `results/`, and code review by a second member on each PR.
- Every member has read and can explain the submitted code.
