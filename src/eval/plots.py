"""Comparison plots generated ONLY from run records. Owner: Member 2."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _names(df: pd.DataFrame) -> list[str]:
    return [f"{m}\n{v}" for m, v in zip(df["model"], df["variant"])]


def comparison_plots(exps: pd.DataFrame, out: Path) -> None:
    df = exps.sort_values(["model", "variant"]).reset_index(drop=True)
    x = np.arange(len(df))
    names = _names(df)

    if "cv.macro_f1_mean" in df:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x, df["cv.macro_f1_mean"], yerr=df.get("cv.macro_f1_std"), capsize=4, color="#4472c4")
        ax.set_xticks(x, names)
        ax.set_ylabel("CV Macro F1 (5-fold GroupKFold, train)")
        ax.set_title("Cross-validation Macro F1 by model")
        fig.tight_layout()
        fig.savefig(out / "cv_macro_f1.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.38
    ax.bar(x - w / 2, df["test.top1"], w, label="Top-1 accuracy", color="#4472c4")
    ax.bar(x + w / 2, df["test.macro_f1"], w, label="Macro F1", color="#ed7d31")
    ax.set_xticks(x, names)
    ax.set_ylabel("student_test")
    ax.set_title("Student-test Top-1 accuracy and Macro F1")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "test_top1_macro_f1.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["test.logloss"], color="#a5a5a5")
    ax.set_xticks(x, names)
    ax.set_ylabel("Log loss (student_test, clipped 1e-6)")
    ax.set_title("Log loss by model (lower is better)")
    fig.tight_layout()
    fig.savefig(out / "test_logloss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, df["infer_ms_per_1k"], color="#70ad47")
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_ylabel("Inference time, ms per 1 000 candidate rows (log)")
    ax.set_title("Inference time by model")
    fig.tight_layout()
    fig.savefig(out / "inference_time.png", dpi=150)
    plt.close(fig)


def confusion_plots(conf: dict, out: Path) -> None:
    for name, cm in conf.items():
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        ax.imshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f"{int(val):,}", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{name} (student_test)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"confusion_{name}.png", dpi=150)
        plt.close(fig)
