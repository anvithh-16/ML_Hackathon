# AI-use disclosure (owner: Member 1)

Required by the PathSnap Assessment & Academic Integrity Policy.

- **Tool(s) used:** Claude Code (Anthropic, model Claude Opus 5.5) — used by Member 1.
- **Tool(s) used (Member 3):** ChatGPT (OpenAI, GPT-6) — used to assist with interpreting Member 3 experiment outputs and drafting/editing the Random Forest, HMM/Viterbi and viva documentation.
- **Tool(s) used (Member 2):** Claude Code (Anthropic; Claude Opus 5.5 and Claude Sonnet 5.5 in different parts of the session) — used by Member 2.
- **Tool(s) used (Member 4):** Claude Code (Anthropic, model Claude Opus 5.5) — used by Member 4.
- **What it was used for (Member 1):**
  - setting up the environment, running the Member 1 pipeline on the real data (validation, folds, labels,
    candidates, features, radius sweep, EDA figures/maps, Logistic Regression runs) and creating the git branches and pull requests;
  - reviewing the starter code against its outputs, which found and fixed one inconsistency (the anomaly decision for
    trip 00000002 did not match `exclude_trips_train`, PR #11);
  - first drafts of report sections 01, 02, 03 (Logistic Regression) and 06, using only numbers from `results/`.
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
- **What it was used for (Member 3):**
  - interpreting the committed Random Forest experiment JSON and HMM aggregate/per-trip/decoder-sweep CSV outputs;
  - drafting report and viva documentation from the repository's implementation, configuration and recorded results.
- **What it was used for (Member 4):**
  - setting up the environment, cloning the team repository, installing the Member 4 starter package files and creating
    the branches, commits and pull requests for the skeleton, `feat/model-interface`, `feat/xgb`, `docs/member4`, the
    re-export (PR #27) and `requirements.lock.txt` (PR #28) (committed under Member 4's own name; no AI co-author line);
  - running the tests, a smoke test of the Optuna search and the export on MOCK synthetic data (not reported), the XGBoost
    train-vs-validation check (`results/eval/xgb_train_vs_val.csv`) and all 10 student_test trips through the API;
  - one small code change: in `src/models/xgb.py`, a `random_state` given in the parameters now overrides the seed
    instead of raising a duplicate-argument error;
  - finding that `artifacts/final/hmm.yaml` still had the pre-tuning `uturn_penalty` (2.0 instead of 5.0), so the API's
    HMM Top-1 did not match the offline results; fixed by re-running the export, after which all 10 student_test trips match;
  - first drafts of report sections 03 (XGBoost), 04 and 07 and of `docs/viva/viva_member4.md`, using only numbers from
    files in `results/` and `artifacts/final/manifest.json`.
- **Problems found while checking AI output (Member 4):** the first draft of section 07 quoted the API Top-1 for trip
  00000007 (0.993) from a run made with the old decoder setting; it now states that and gives the value after the
  re-export (0.994). A shell command used while preparing the API branch created empty folders with wrong names; they
  were deleted before anything was committed.
- **What the team did itself:** deciding the plan and work split, choosing the model, reviewing every map and figure,
  deciding on flagged trips, rewriting the report text in our own words, and reviewing and merging every PR.
- **How we verified AI-produced code and text:** unit tests (`tests/`), checks on real data against the expected
  counts (80 / 208 417 train observations, 10 / 19 160 student_test), a nearest-candidate baseline to check metrics,
  every report number traced back to a file in `results/`, and code review by a second member on each PR.
- Every member has read and can explain the submitted code.
