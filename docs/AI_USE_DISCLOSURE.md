# AI-use disclosure (owner: Member 1)

Required by the PathSnap Assessment & Academic Integrity Policy.

- **Tool(s) used:** Claude Code (Anthropic, model Claude Opus 5.5) — used by Member 1.
- **Tool(s) used (Member 3):** ChatGPT (OpenAI, GPT-6) — used to assist with interpreting Member 3 experiment outputs and drafting/editing the Random Forest, HMM/Viterbi and viva documentation.
- **What it was used for (Member 1):**
  - setting up the environment, running the Member 1 pipeline on the real data (validation, folds, labels,
    candidates, features, radius sweep, EDA figures/maps, Logistic Regression runs) and creating the git branches and pull requests;
  - reviewing the starter code against its outputs, which found and fixed one inconsistency (the anomaly decision for
    trip 00000002 did not match `exclude_trips_train`, PR #11);
  - first drafts of report sections 01, 02, 03 (Logistic Regression) and 06, using only numbers from `results/`.
- **What it was used for (Member 3):**
  - interpreting the committed Random Forest experiment JSON and HMM aggregate/per-trip/decoder-sweep CSV outputs;
  - drafting report and viva documentation from the repository's implementation, configuration and recorded results.
- **What the team did itself:** deciding the plan and work split, choosing the model, reviewing every map and figure,
  deciding on flagged trips, rewriting the report text in our own words, and reviewing and merging every PR.
- **How we verified AI-produced code and text:** unit tests (`tests/`), checks on real data against the expected
  counts (80 / 208 417 train observations, 10 / 19 160 student_test), a nearest-candidate baseline to check metrics,
  every report number traced back to a file in `results/`, and code review by a second member on each PR.
- Every member has read and can explain the submitted code.
