"""Assemble docs/report/PathSnap_Technical_Report.pdf from the section files and results/. Owner: Member 1.

Run from the repo root:  python docs/report/build_report.py
Sections still in template form (they contain the organiser banner) are NOT pasted; instead the report shows a
"pending" box plus the real numbers generated from results/. Re-run after teammates push their sections.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parents[2]
SEC = ROOT / "docs" / "report" / "sections"
OUT = ROOT / "docs" / "report" / "PathSnap_Technical_Report.pdf"
BANNER = "Do not paste numbers you did not produce"

# ---------------------------------------------------------------- fonts / styles
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(_fd / f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

BASE = ParagraphStyle("base", fontName="DV", fontSize=9, leading=12.4, spaceAfter=4)
H1 = ParagraphStyle("h1", parent=BASE, fontName="DV-B", fontSize=17, leading=21, spaceBefore=6, spaceAfter=8,
                    textColor=colors.HexColor("#1f3864"))
H2 = ParagraphStyle("h2", parent=BASE, fontName="DV-B", fontSize=12.5, leading=16, spaceBefore=9, spaceAfter=4,
                    textColor=colors.HexColor("#2f5597"))
H3 = ParagraphStyle("h3", parent=BASE, fontName="DV-B", fontSize=10.3, leading=13.5, spaceBefore=6, spaceAfter=3)
QUOTE = ParagraphStyle("q", parent=BASE, fontName="DV-I", textColor=colors.HexColor("#555555"), leftIndent=8)
BUL = ParagraphStyle("b", parent=BASE, leftIndent=14, bulletIndent=4, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=BASE, fontSize=7.3, leading=9, spaceAfter=0)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName="DV-B")
CODE = ParagraphStyle("code", fontName="DVM", fontSize=7.2, leading=9, leftIndent=6, backColor=colors.HexColor("#f2f2f2"))
CAP = ParagraphStyle("cap", parent=BASE, fontName="DV-I", fontSize=7.8, leading=10, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"))
PEND = ParagraphStyle("pend", parent=BASE, backColor=colors.HexColor("#fff4cc"), borderColor=colors.HexColor("#d9a900"),
                      borderWidth=0.8, borderPadding=5, spaceBefore=4, spaceAfter=8)
W = A4[0] - 36 * mm


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t: str) -> str:
    t = esc(t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"`([^`]+)`", r'<font name="DVM" size="7.8">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    return t


def image(path: Path, caption: str | None = None, width: float = W * 0.92, maxh: float = 105 * mm):
    if not path.exists():
        return [Paragraph(f"[missing figure: {esc(str(path.relative_to(ROOT)))}]", CAP)]
    from PIL import Image as PI
    w, h = PI.open(path).size
    sc = min(width / w, maxh / h)
    items = [Image(str(path), width=w * sc, height=h * sc)]
    items.append(Paragraph(esc(caption or path.name), CAP))
    return [KeepTogether(items), Spacer(1, 4)]


def table(rows: list[list[str]], header: bool = True, widths=None):
    data = [[Paragraph(inline(c), CELLH if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    n = max(len(r) for r in rows)
    for r in data:
        r += [Paragraph("", CELL)] * (n - len(r))
    t = Table(data, colWidths=widths or [W / n] * n, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")))
    t.setStyle(TableStyle(st))
    return [t, Spacer(1, 6)]


def md_to_flow(text: str, base_dir: Path, shift: int = 0):
    """Small Markdown subset -> reportlab flowables."""
    flow, lines, i = [], text.splitlines(), 0
    para: list[str] = []

    def flush():
        if para:
            flow.append(Paragraph(inline(" ".join(para)), BASE))
            para.clear()

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            flow.append(Preformatted("\n".join(buf), CODE))
            i += 1
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            flush()
            flow += image((base_dir / m.group(2)).resolve(), m.group(1) or None)
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?[\s:|-]+\|?$", lines[i + 1].strip()):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                if not re.match(r"^\|?[\s:|-]+\|?$", lines[i].strip()):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            flow += table(rows)
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            lvl = min(len(m.group(1)) + shift, 4)
            flow.append(Paragraph(inline(m.group(2)), {1: H1, 2: H2}.get(lvl, H3)))
        elif s.startswith(">"):
            flush()
            flow.append(Paragraph(inline(s.lstrip("> ")), QUOTE))
        elif re.match(r"^(\s*)([-*]|\d+\.)\s+", ln):
            flush()
            m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", ln)
            ind = len(m.group(1)) // 2
            bullet = "•" if m.group(2) in "-*" else m.group(2)
            st = ParagraphStyle("bi", parent=BUL, leftIndent=14 + 12 * ind, bulletIndent=4 + 12 * ind)
            item = [m.group(3).strip()]
            while (i + 1 < len(lines) and lines[i + 1].strip() and lines[i + 1].startswith((" ", "\t"))
                   and not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i + 1])):
                i += 1
                item.append(lines[i].strip())
            flow.append(Paragraph(inline(" ".join(item)), st, bulletText=bullet))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return flow


# ---------------------------------------------------------------- results-derived content
def load_runs() -> dict[str, dict]:
    runs = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "experiments" / "*.json"))):
        r = json.load(open(f))
        if not r.get("synthetic"):
            runs[f"{r['model']}__{r['variant']}"] = r  # later timestamps overwrite earlier ones
    return runs


def model_results_table(key: str, runs: dict):
    r = runs.get(key)
    if not r:
        return [Paragraph(f"No run record found for {key}.", BASE)]
    pf = r["cv"]["per_fold"]
    import statistics as st
    ms = lambda k: f"{st.mean(p[k] for p in pf):.4f} ± {st.pstdev(p[k] for p in pf):.4f}"
    t = r["test"]
    rows = [["Quantity", "Value", "Source"],
            ["Configuration", esc(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["params"].items()})), "run record"],
            ["Training data", f"{r['n_train_trips']} trips, {r['n_train_rows']:,} rows (stride {r['obs_stride']})", "run record"],
            ["CV Top-1 (5 folds, mean ± std)", ms("top1"), "run record"],
            ["CV Macro F1", ms("macro_f1"), "run record"],
            ["CV log loss", ms("logloss"), "run record"],
            ["student_test Top-1", f"{t['top1']:.4f}", "run record"],
            ["student_test Macro F1", f"{t['macro_f1']:.4f}", "run record"],
            ["student_test log loss", f"{t['logloss']:.4f}", "run record"],
            ["student_test precision / recall (positive class)", f"{t['precision']:.4f} / {t['recall']:.4f}", "run record"],
            ["Training time / inference", f"{r['train_time_s']:.2f} s / {r['infer_ms_per_1k']:.3f} ms per 1k rows", "run record"],
            ["Record", f"results/experiments/{r['model']}__{r['variant']}__{r['timestamp']}.json", ""]]
    return table(rows, widths=[W * 0.34, W * 0.5, W * 0.16])


def csv_rows(path: Path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def tuning_block(runs):
    f = ROOT / "results" / "tuning"
    out = []
    bp = f / "xgb_best_params.json"
    if bp.exists():
        d = json.load(open(bp))
        out.append(Paragraph(f"Optuna TPE, <b>{d['n_trials']}</b> trials, objective <b>{esc(d['objective'])}</b> "
                             f"(mean group-aware CV Macro F1 on the tuning subsample). Best trial {d['best_value']:.4f} vs untuned "
                             f"trial 0 {d['untuned_value']:.4f}.", BASE))
        out += table([["Hyperparameter", "Best value"]] + [[k, f"{v:.4g}" if isinstance(v, float) else str(v)] for k, v in d["params"].items()],
                     widths=[W * 0.4, W * 0.3])
    rows = csv_rows(f / "tuned_vs_untuned.csv")
    if rows:
        keep = ["Model", "CV Macro F1", "Top-1 Accuracy", "Student-Test Macro F1", "Log Loss", "Inference time (ms / 1k rows)"]
        out += table([keep] + [[(f"{float(r[k]):.4f}" if k in ("Top-1 Accuracy", "Student-Test Macro F1", "Log Loss", "Inference time (ms / 1k rows)") else r[k]) for k in keep] for r in rows])
    out += image(f / "xgb_history.png", "Optuna optimisation history (results/tuning/xgb_history.png)")
    return out


def hmm_block():
    f = ROOT / "results" / "hmm"
    out = []
    y = f / "best_decoder.yaml"
    if y.exists():
        out.append(Paragraph("Decoder parameters (tuned on training out-of-fold scores, <font name=\"DVM\">results/hmm/best_decoder.yaml</font>):", BASE))
        out.append(Preformatted(y.read_text().strip(), CODE))
        out.append(Spacer(1, 4))
    rows = csv_rows(f / "clf_vs_hmm.csv")
    if rows:
        cols = [("scores", "scores"), ("method", "method"), ("top1", "Top-1"), ("route_continuity", "continuity"),
                ("route_precision", "route prec."), ("route_recall", "route recall"), ("n_breaks", "breaks/trip"),
                ("frac_unmatched", "unmatched")]
        out.append(Paragraph("student_test, 10 trips (<font name=\"DVM\">results/hmm/clf_vs_hmm.csv</font>):", BASE))
        out += table([[c[1] for c in cols]] + [[(f"{float(r[k]):.4f}" if k not in ("scores", "method") else r[k]) for k, _ in cols] for r in rows])
    out += image(f / "figures" / "clf_vs_hmm_top1.png", "Classifier-only vs classifier + HMM (results/hmm/figures)")
    sw = csv_rows(f / "decoder_sweep.csv")
    if sw:
        out.append(Paragraph(f"Decoder sweep: {len(sw)} (β, U-turn penalty) combinations on 20 training trips "
                             f"(<font name=\"DVM\">results/hmm/decoder_sweep.csv</font>); best HMM Top-1 "
                             f"{max(float(r['hmm_top1_mean']) for r in sw):.4f}.", BASE))
    return out


def pending_box(sec_file: str, owner: str, what: str):
    return Paragraph(f"<b>Section pending ({owner}).</b> The written answers in <font name=\"DVM\">{sec_file}</font> "
                     f"are still the unfilled template ({what}). The numbers below were generated directly from "
                     f"<font name=\"DVM\">results/</font>; the owner's analysis replaces this box once their section is pushed.", PEND)


# ---------------------------------------------------------------- document assembly
def sec_text(name: str):
    p = SEC / name
    return p.read_text() if p.exists() else None


def strip_title(t: str) -> str:
    return re.sub(r"^#\s+.*\n", "", t, count=1)


def build() -> Path:
    runs = load_runs()
    story: list = []
    story += [Spacer(1, 55 * mm), Paragraph("PathSnap", ParagraphStyle("t", parent=H1, fontSize=34, leading=40, alignment=TA_CENTER)),
              Paragraph("GPS map matching and route reconstruction", ParagraphStyle("s", parent=H2, fontSize=15, alignment=TA_CENTER)),
              Spacer(1, 10 * mm),
              Paragraph("Technical Report — UE24CS352A Machine Learning Hackathon", ParagraphStyle("c", parent=BASE, fontSize=11, alignment=TA_CENTER)),
              Spacer(1, 6 * mm),
              Paragraph("Repository: github.com/anvithh-16/ML_Hackathon", ParagraphStyle("c2", parent=BASE, fontSize=10, alignment=TA_CENTER)),
              Spacer(1, 40 * mm),
              Paragraph("All numbers in this report are read from files under <font name=\"DVM\">results/</font> produced by the scripts in the "
                        "repository (see README: Reproduce everything). Sections marked <i>pending</i> are still waiting for the "
                        "owner's written analysis.", ParagraphStyle("n", parent=QUOTE, alignment=TA_CENTER)),
              PageBreak()]
    pend: list[str] = []

    def add_section(title_flow, fname: str, owner: str, what: str, extra=None, shift=0):
        t = sec_text(fname)
        if t is None:
            story.append(Paragraph(f"[missing {fname}]", PEND))
            pend.append(fname)
            return
        if BANNER in t:
            pend.append(fname)
            story.append(title_flow)
            story.append(pending_box(f"docs/report/sections/{fname}", owner, what))
        else:
            story.append(title_flow)
            story.extend(md_to_flow(strip_title(t), SEC, shift))
        if extra:
            story.extend(extra() if callable(extra) else extra)

    # ----- 2
    story.append(Paragraph("2. Dataset understanding and exploratory data analysis", H1))
    story.extend(md_to_flow(strip_title(sec_text("01_dataset_eda.md")), SEC, shift=0))
    eda = ROOT / "results" / "eda" / "figures"
    cap = {"sampling_interval.png": "Sampling interval (train + student_test)", "obs_vs_route_len.png": "GPS rows vs route entries (train + student_test)",
           "gps_to_route_distance.png": "GPS-to-route distance (train, excluded trips removed)", "network_size.png": "Road network size (train + student_test)",
           "obs_per_trip.png": "Observations per trip (train + student_test)", "candidates_per_obs.png": "Candidates per observation (train, radius 50 m)",
           "trajectory_train_00000001.png": "GPS trajectory, train trip 00000001", "trajectory_student_test_00000000.png": "GPS trajectory, student_test trip 00000000"}
    for k, v in cap.items():
        story += image(eda / k, v, width=W * 0.72, maxh=80 * mm)
    story.append(Paragraph("Interactive Folium maps (GPS, road network, reference route; layer control and legend) are committed under "
                           "<font name=\"DVM\">results/eda/maps/</font>; static overlays of the same data are shown above and in §2.5.", BASE))
    story.append(PageBreak())
    story.append(Paragraph("2.5 Labels, candidates and features", H1))
    story.extend(md_to_flow(strip_title(sec_text("02_features_labels.md")), SEC, shift=0))
    story.append(PageBreak())

    # ----- 3
    story.append(Paragraph("3. Model experimentation and comparison", H1))
    t = sec_text("03_comparison.md")
    story.extend(md_to_flow(strip_title(t), SEC, shift=1))
    story.append(PageBreak())
    story.append(Paragraph("3.2 Per-model results (compulsory questions A–C)", H1))
    models = [("03_model_logreg.md", "logreg__default", "Member 1", "Logistic Regression", "confusion_logreg__default.png"),
              ("03_model_knn.md", "knn__default", "Member 2", "K-Nearest Neighbours", "confusion_knn__default.png"),
              ("03_model_dtree.md", "dtree__default", "Member 2", "Decision Tree", "confusion_dtree__default.png"),
              ("03_model_rforest.md", "rforest__default", "Member 3", "Random Forest", "confusion_rforest__default.png"),
              ("03_model_xgb.md", "xgb__default", "Member 4", "XGBoost", "confusion_xgb__default.png")]
    for fname, key, owner, title, conf in models:
        t = sec_text(fname)
        if t and BANNER not in t:
            story.extend(md_to_flow(t, SEC, shift=1))
        else:
            pend.append(fname)
            story.append(Paragraph(f"3.2 {title}", H2))
            story.append(pending_box(f"docs/report/sections/{fname}", owner, "model understanding, configuration and over/under-fitting analysis"))
        story.extend(model_results_table(key, runs))
        story += image(ROOT / "results" / "figures" / conf, f"Confusion matrix, {title}, student_test", width=W * 0.42, maxh=60 * mm)
    story.append(Paragraph("3.3 Model comparison figures", H2))
    for f_, c_ in [("cv_macro_f1.png", "Cross-validation Macro F1 (5 group-aware folds, train)"),
                   ("test_top1_macro_f1.png", "student_test Top-1 accuracy and Macro F1"),
                   ("test_logloss.png", "student_test log loss"), ("inference_time.png", "Inference time (log scale)")]:
        story += image(ROOT / "results" / "figures" / f_, c_, width=W * 0.7, maxh=75 * mm)
    story.append(Paragraph("Neural-network-style training and validation loss curves are not shown: none of the five models produces an epoch-wise "
                           "training history, and we did not fabricate any.", BASE))
    story.append(PageBreak())

    # ----- 4
    story.append(Paragraph("4. Bayesian hyperparameter optimisation (XGBoost)", H1))
    t = sec_text("04_bayesian_optimisation.md")
    if t and BANNER not in t:
        story.extend(md_to_flow(strip_title(t), SEC, shift=1))
    else:
        pend.append("04_bayesian_optimisation.md")
        story.append(pending_box("docs/report/sections/04_bayesian_optimisation.md", "Member 4", "search space, ranges, interpretation"))
    story.extend(tuning_block(runs))
    story.append(PageBreak())

    # ----- 5
    story.append(Paragraph("5. HMM and Viterbi route reconstruction", H1))
    t = sec_text("05_hmm_viterbi.md")
    if t and BANNER not in t:
        story.extend(md_to_flow(strip_title(t), SEC, shift=1))
    else:
        pend.append("05_hmm_viterbi.md")
        story.append(pending_box("docs/report/sections/05_hmm_viterbi.md", "Member 3", "hidden states, emissions, transitions, gap handling, where the HMM did not help"))
    story.extend(hmm_block())
    story.append(PageBreak())

    # ----- 6
    story.append(Paragraph("6. Geographical visualisation (Folium)", H1))
    story.extend(md_to_flow(strip_title(sec_text("06_folium_maps.md")), SEC, shift=1))
    story += image(ROOT / "results" / "labels" / "figures" / "align_train_00000001.png", "Normal trip: train 00000001, all points labelled (static overlay of the committed Folium map)")
    story += image(ROOT / "results" / "labels" / "figures" / "align_train_00000081.png", "Flagged trip: train 00000081 (red = label −1)")
    story.append(PageBreak())

    # ----- 7
    story.append(Paragraph("7. API, user interface and deployment", H1))
    for fname, owner, what in [("07_api_deployment.md", "Member 4", "API design, request flow, deployment evidence"),
                               ("07b_ui.md", "Member 2", "UI design and screenshots")]:
        t = sec_text(fname)
        if t and BANNER not in t:
            story.extend(md_to_flow(strip_title(t), SEC, shift=1))
        else:
            pend.append(fname)
            story.append(pending_box(f"docs/report/sections/{fname}", owner, what))
    story.append(Paragraph("Verified locally on the final merged code: the full test suite passes (42 tests) and all 10 student_test trips "
                           "return HTTP 200 from <font name=\"DVM\">POST /predict/{trip_id}</font> with the exported XGBoost model "
                           "(<font name=\"DVM\">/health</font> reports mode “model”); the request flow, evaluator rehearsal and screenshots are "
                           "described by Members 2 and 4 above. Run commands: <font name=\"DVM\">docs/DEPLOYMENT.md</font>.", BASE))
    story.append(PageBreak())

    # ----- appendix
    story.append(Paragraph("Appendix A. AI-use disclosure", H1))
    story.extend(md_to_flow(strip_title(sec_text("../../AI_USE_DISCLOSURE.md".replace("../../", "")) or (ROOT / "docs" / "AI_USE_DISCLOSURE.md").read_text()), SEC, 1))

    def footer(c, d):
        c.saveState()
        c.setFont("DV", 7.5)
        c.setFillColor(colors.grey)
        c.drawString(18 * mm, 10 * mm, "PathSnap — technical report")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=17 * mm,
                            title="PathSnap Technical Report", author="PathSnap team")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=footer)
    print("wrote", OUT.relative_to(ROOT), "| pending sections:", sorted(set(pend)) or "none")
    return OUT


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
