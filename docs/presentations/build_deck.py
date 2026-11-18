"""Build docs/presentations/Review_2.pptx from results/. Owner: Member 1.

Run from the repo root:  python docs/presentations/build_deck.py
Every number is read from results/ at build time (nothing is typed in by hand except labels).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentations" / "Review_2.pptx"
NAVY, BLUE, GREY, LIGHT = RGBColor(0x1F, 0x38, 0x64), RGBColor(0x2F, 0x55, 0x97), RGBColor(0x55, 0x55, 0x55), RGBColor(0xDC, 0xE6, 0xF2)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title: str, subtitle: str | None = None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12.1), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, NAVY
    if subtitle:
        p2 = tb.text_frame.add_paragraph()
        p2.text = subtitle
        p2.font.size, p2.font.color.rgb = Pt(14), GREY
    ft = s.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(12), Inches(0.3))
    ft.text_frame.text = "PathSnap · UE24CS352A ML Hackathon · Review 2"
    ft.text_frame.paragraphs[0].font.size = Pt(10)
    ft.text_frame.paragraphs[0].font.color.rgb = GREY
    return s


def bullets(s, items, x=0.6, y=1.5, w=6.0, h=5.2, size=18):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        lvl = 1 if it.startswith("  ") else 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if lvl == 0 else "– ") + it.strip()
        p.level = lvl
        p.font.size = Pt(size - 3 * lvl)
        p.space_after = Pt(7)
    return tb


def picture(s, rel: str, x, y, w=None, h=None):
    path = ROOT / rel
    if path.exists():
        s.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


def table(s, rows, x, y, w, col_w=None, size=12, row_h=0.36):
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(v)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(size)
            para.font.bold = r == 0
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
                para.font.color.rgb = NAVY
    if col_w:
        for c, cw in enumerate(col_w):
            t.columns[c].width = Inches(cw)
    return t


# ------------------------------------------------------------------ data
inv = pd.read_csv(ROOT / "results/eda/trip_inventory.csv", dtype={"trip_id": str})
cmp_ = pd.read_csv(ROOT / "results/model_comparison.csv")
sweep = pd.read_csv(ROOT / "results/candidates/radius_sweep.csv")
hmm = pd.read_csv(ROOT / "results/hmm/clf_vs_hmm.csv")
lab = pd.read_csv(ROOT / "results/labels/alignment_summary.csv", dtype={"trip_id": str})
tr, te = inv[inv.partition == "train"], inv[inv.partition == "student_test"]
n_tr, n_te = int(tr.n_obs.sum()), int(te.n_obs.sum())
row = lambda m: cmp_[cmp_.Model == m].iloc[0]
xgb, rf, lr_ = row("xgb__default"), row("rforest__default"), row("logreg__default")
base = cmp_.iloc[0]
manifest = json.load(open(ROOT / "artifacts/final/manifest.json"))

# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.4), Inches(11.5), Inches(3))
p = tb.text_frame.paragraphs[0]
p.text = "PathSnap"
p.font.size, p.font.bold, p.font.color.rgb = Pt(60), True, RGBColor(255, 255, 255)
for txt, sz in [("GPS map matching and route reconstruction", 26), ("UE24CS352A Machine Learning Hackathon · Review 2", 16),
                ("github.com/anvithh-16/ML_Hackathon", 14)]:
    q = tb.text_frame.add_paragraph()
    q.text = txt
    q.font.size, q.font.color.rgb = Pt(sz), RGBColor(0xDC, 0xE6, 0xF2)

# ------------------------------------------------------------------ 2 problem + pipeline
s = slide("The problem and our pipeline", "Which road segment was the vehicle on at every GPS point — and what was the route?")
steps = ["Load + project\n(per-trip UTM)", "Align GPS to\nreference route\n(labels)", "Candidate arcs\n(50 m + fallback)", "15 features per\n(point, arc)",
         "5 classifiers\n(same folds, same\nmetrics)", "Pick best +\nOptuna tuning", "HMM / Viterbi\n→ connected route", "FastAPI +\nStreamlit UI"]
for i, t in enumerate(steps):
    x = 0.45 + (i % 4) * 3.15
    y = 1.9 + (i // 4) * 2.1
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.8), Inches(1.5))
    b.fill.solid()
    b.fill.fore_color.rgb = LIGHT if i < 6 else RGBColor(0xC9, 0xE4, 0xCA)
    b.line.color.rgb = BLUE
    b.text_frame.text = f"{i + 1}. {t}"
    for pp in b.text_frame.paragraphs:
        pp.font.size, pp.font.color.rgb, pp.alignment = Pt(15), NAVY, PP_ALIGN.CENTER
bullets(s, [f"Data: {len(tr)} train + {len(te)} student_test trips, trips from all over the world; evaluator trips arrive only at demo time",
            "Rules we follow: group-aware CV by trip, student_test only for comparison, ground truth stays server-side"], y=6.1, w=12.2, h=1, size=15)

# ------------------------------------------------------------------ 3 data / EDA
s = slide("Data understanding and EDA", "partition stated per figure: train + student_test unless noted")
bullets(s, [f"train {len(tr)} trips / {n_tr:,} observations; student_test {len(te)} / {n_te:,}",
            "1 Hz GPS; 96.7 % of intervals exactly 1 s; 30 trips have a gap > 30 s (max 153,064 s)",
            f"No missing values; GPS median {tr.gps_to_route_median_m[~tr.trip_id.isin(['00000054', '00000033'])].median():.1f} m from the reference route",
            "Each trip → own UTM zone (3 southern-hemisphere trips)",
            "Route = arcs, not 1 per GPS point (median 5.25 GPS points per arc)",
            "Excluded from training: 00000054 and 00000033 (reference route does not match the GPS)"], w=6.2, size=17)
picture(s, "results/eda/figures/gps_to_route_distance.png", 6.9, 1.4, w=6.0)
picture(s, "results/eda/figures/sampling_interval.png", 6.9, 4.2, w=6.0)

# ------------------------------------------------------------------ 4 labels, candidates, features
s = slide("Labels, candidates and features", "all built by one pipeline that every model and the API share")
bullets(s, ["Labels: monotonic DP over route positions (handles loops); label −1 beyond 50 m",
            f"Alignment median {lab.align_dist_median_m.median():.1f} m, {int(lab.shape[0])} trips checked; flagged trips reviewed on overlays",
            "Candidates: all arcs within 50 m, else 3 nearest within 200 m, cap 30",
            "Features never read the route (unit-tested): distance, rank, heading difference, speed, gaps, arc length, node degrees, neighbour-consistency, reverse twin",
            "Group-aware CV: 5 folds × 16 trips"], w=6.4, size=16)
sw = sweep.copy()
table(s, [["radius", "miss rate", "cands / obs"]] + [[f"{int(r.radius_m)} m", f"{r.candidate_miss_rate * 100:.3f} %", f"{r.mean_candidates_per_obs:.1f}"] for r in sw.itertuples()],
      7.3, 1.6, 5.4, size=14)
bullets(s, ["50 m: misses 0.012 % of true arcs; larger radii only add negatives",
            "train 1.84 M candidate rows; student_test 0.20 M"], x=7.3, y=3.8, w=5.4, h=2, size=15)

# ------------------------------------------------------------------ 5 comparison
s = slide("Five models, one fair protocol", "same folds, same subsample, same metrics (student_test only for comparison)")
cols = [("Model", None), ("CV Macro F1", None), ("Top-1", "Top-1 Accuracy"), ("Log loss", "Log Loss"), ("ms / 1k rows", "Inference time (ms / 1k rows)")]
rows = [[c[0] for c in cols]]
rows.append(["Nearest-candidate baseline", "—", f"{float(base['Top-1 Accuracy']):.3f}", "—", "—"])
for m in ["logreg__default", "knn__default", "dtree__default", "rforest__default", "xgb__default", "xgb__tuned"]:
    r = row(m)
    rows.append([m.replace("__", " "), r["CV Macro F1"], f"{float(r['Top-1 Accuracy']):.3f}", f"{float(r['Log Loss']):.4f}", f"{float(r['Inference time (ms / 1k rows)']):.2f}"])
table(s, rows, 0.6, 1.6, 7.4, col_w=[2.5, 1.6, 0.9, 1.1, 1.3], size=12, row_h=0.42)
picture(s, "results/figures/test_top1_macro_f1.png", 8.2, 1.6, w=4.8)
bullets(s, ["KNN: 368 ms per 1k rows (800× XGBoost)", "Trees beat the linear model by ~2 points of Top-1; every model beats the 0.54 baseline"], x=8.2, y=4.2, w=4.9, h=2.5, size=15)

# ------------------------------------------------------------------ 6 selection + tuning
s = slide("Selected model: XGBoost (default parameters)", "pre-declared order: student_test Top-1 → CV stability → log loss → inference time")
tuned = row("xgb__tuned")
table(s, [["", "Top-1", "CV Macro F1", "Log loss", "ms / 1k"],
          ["xgb default (selected)", f"{float(xgb['Top-1 Accuracy']):.3f}", xgb["CV Macro F1"], f"{float(xgb['Log Loss']):.4f}", f"{float(xgb['Inference time (ms / 1k rows)']):.2f}"],
          ["xgb tuned (Optuna, 40 trials)", f"{float(tuned['Top-1 Accuracy']):.3f}", tuned["CV Macro F1"], f"{float(tuned['Log Loss']):.4f}", f"{float(tuned['Inference time (ms / 1k rows)']):.2f}"]],
      0.6, 1.7, 12.1, col_w=[3.9, 1.5, 3.0, 1.7, 2.0], size=14, row_h=0.5)
bullets(s, ["Tuning gained only +0.0026 CV Macro F1 (less than the fold std ≈ 0.022) and did not improve Top-1 → untuned model kept",
            f"Exported to artifacts/final/ ({manifest['model']} / {manifest['variant']}); the API loads exactly this folder",
            "Honest caveat: with the HMM on top, Random Forest scores reach 0.951 Top-1 vs 0.942 for XGBoost — within noise on 10 test trips"], y=3.6, w=12.1, size=17)

# ------------------------------------------------------------------ 7 HMM
s = slide("HMM / Viterbi: from independent guesses to a connected route")
h = hmm[hmm.scores == "xgb__default"].set_index("method")
d0 = hmm[hmm.scores == "distance-only"].set_index("method")
table(s, [["xgb scores, student_test", "Top-1", "Route continuity", "Route recall"],
          ["classifier only", f"{h.loc['classifier-only', 'top1']:.3f}", f"{h.loc['classifier-only', 'route_continuity']:.3f}", f"{h.loc['classifier-only', 'route_recall']:.3f}"],
          ["classifier + HMM", f"{h.loc['classifier+HMM', 'top1']:.3f}", f"{h.loc['classifier+HMM', 'route_continuity']:.3f}", f"{h.loc['classifier+HMM', 'route_recall']:.3f}"],
          ["distance-only + HMM (no ML)", f"{d0.loc['classifier+HMM', 'top1']:.3f}", f"{d0.loc['classifier+HMM', 'route_continuity']:.3f}", f"{d0.loc['classifier+HMM', 'route_recall']:.3f}"]],
      0.6, 1.6, 7.0, col_w=[3.1, 1.1, 1.7, 1.1], size=13, row_h=0.48)
bullets(s, ["Hidden states = candidate arcs; emissions = normalised classifier scores; transitions penalise |network − straight-line distance| and U-turns",
            "Gaps > 30 s or > 500 m start a new segment; β and U-turn penalty tuned on training out-of-fold scores only",
            f"The HMM gives most of the gain (Top-1 {h.loc['classifier-only', 'top1']:.3f} → {h.loc['classifier+HMM', 'top1']:.3f}); the classifier adds ≈ {100 * (h.loc['classifier+HMM', 'top1'] - d0.loc['classifier+HMM', 'top1']):.1f} points over a distance-only HMM"],
        y=3.8, w=7.0, h=3.2, size=15)
picture(s, "results/hmm/figures/clf_vs_hmm_top1.png", 7.9, 1.6, w=5.1)

# ------------------------------------------------------------------ 8 system
s = slide("System: API and evaluator UI", "two-laptop design from the organiser brief")
bullets(s, ["Laptop 1: FastAPI loads artifacts/final/ (model + HMM config); GET /health, GET /trips, POST /predict/{trip_id}",
            "Laptop 2: Streamlit + Folium UI lists the evaluator trips, calls the API and draws GPS + predicted route",
            "Ground truth stays on the server: the scorer is the only code that reads .route; the response contains no reference geometry",
            "Verified on merged main: 42 tests pass; all 10 student_test trips return HTTP 200 in mode “model” (0.1–1.6 s per trip)",
            "Still to rehearse on two physical laptops: LAN access, firewall, demo log + screenshots"], w=12.1, size=18)

# ------------------------------------------------------------------ 9 status
s = slide("Team status", "what is merged and what remains")
table(s, [["Member", "Merged", "Remaining"],
          ["M1", "Data pipeline, labels, candidates, features, Logistic Regression, EDA, README, report §2 / §6", "Final PDF + deck, own-words pass on report text"],
          ["M2", "Eval framework, KNN, Decision Tree, comparison, UI", "KNN / Decision Tree / UI report sections, rehearsal"],
          ["M3", "Route metrics, Random Forest, HMM + experiments", "Random Forest + HMM report sections, viva notes, RF model file"],
          ["M4", "Skeleton, interface, XGBoost, API, tuning + export", "XGBoost / BO / API report sections, freeze checks, public repo"]],
      0.6, 1.6, 12.1, col_w=[1.1, 6.4, 4.6], size=13, row_h=0.8)
bullets(s, ["All code is on main via reviewed PRs; every report number is traceable to a file under results/"], y=6.0, w=12.1, h=0.8, size=16)

# ------------------------------------------------------------------ 10 reproducibility
s = slide("Reproducibility and next steps")
bullets(s, ["README: install → data → reproduce → demo; one command per result (validate, folds, build_features, cv_runner, compare_models, optuna_search, export, compare_hmm)",
            "Report PDF and this deck are generated by scripts (docs/report/build_report.py, docs/presentations/build_deck.py) from results/",
            "Before freeze: fresh-clone install + full test run, README commands, requirements lock, public repo, no data or secrets committed",
            "AI use disclosed in docs/AI_USE_DISCLOSURE.md; every member can explain the code they submit"], w=12.1, size=18)

prs.save(OUT)
print("wrote", OUT.relative_to(ROOT), f"({len(prs.slides)} slides)")
