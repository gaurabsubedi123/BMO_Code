"""
Regenerates the five figures used by the TRCI manuscript and its supplement.

    python3 make_figures.py

Reads the re-run outputs (BMO_RERUN_DIR) and writes PDF (for LaTeX) and 300 dpi
PNG into figures/ (BMO_FIG_OUT). With BMO_SUBMISSION_DIR set, the two main-text
figures are also copied there as Figure1.pdf and Figure2.pdf.

    BMO_RERUN_DIR=/path/to/rerun_results python3 analysis/make_figures.py

Style rules for this paper: plain ASCII text in every label, no in-plot titles
(the caption does that job), legends outside the data, marker shape as well as
colour for tier so the figures survive greyscale printing.
"""
import itertools
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import os

HERE = Path(__file__).resolve().parent
# The pipeline's output directory (outputs/, tables/). The annotated corpus it
# contains is not released, so these figures regenerate only where it exists.
RUN = Path(os.environ.get("BMO_RERUN_DIR", HERE.parent.parent.parent / "rerun_results"))
OUT = Path(os.environ.get("BMO_FIG_OUT", HERE.parent / "figures"))
# BiomedBERT tokenizer, used only for the token-length panel of Figure S3.
TOKENIZER = Path(os.environ.get("BMO_TOKENIZER",
                                HERE.parent.parent.parent / "deployment_model" / "bmo-transformer" / "tokenizer"))
# Written by analysis/recall_by_modality.py; the committed copy is the default.
MODALITY_CSV = Path(os.environ.get(
    "BMO_MODALITY_CSV", HERE.parent / "results" / "rerun_2026-08-28" / "recall_by_modality_oof.csv"))

# Arial if the Windows fonts are reachable, otherwise matplotlib's default sans.
for f in ("arial.ttf", "arialbd.ttf", "ariali.ttf"):
    p = Path("/mnt/c/Windows/Fonts") / f
    if p.exists():
        font_manager.fontManager.addfont(str(p))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.7,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

# Okabe-Ito order, fixed per tier. Shape carries the same information.
TIERS = {
    "Transformer":  dict(label="Fine-tuned encoder",   color="#D55E00", marker="D", ms=5.2),
    "BioWordVec":   dict(label="TF-IDF + BioWordVec",  color="#009E73", marker="^", ms=5.8),
    "FastText":     dict(label="TF-IDF + FastText",    color="#56B4E9", marker="s", ms=5.0),
    "Classical":    dict(label="TF-IDF only",          color="#0072B2", marker="o", ms=5.4),
    "Prompt-based": dict(label="Prompted, no task training", color="#CC79A7", marker="v", ms=5.8),
}


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


def display_name(s):
    """Prediction files key classical systems as 'Tier | Model'."""
    s = s.replace("Classical | ", "TF-IDF + ").replace("FastText | ", "FastText + ")
    s = s.replace("BioWordVec | ", "BioWordVec + ")
    return s.replace("zero-shot", "").replace("  ", " ").replace(" (", " (").strip()


def tidy(s):
    """Shorter, plainer labels for axes."""
    s = s.replace(" (tuned threshold)", ", tuned").replace(" zero-shot", "")
    s = s.replace("Logistic Regression", "logistic regression")
    s = s.replace("Random Forest", "random forest").replace("Decision Tree", "decision tree")
    s = s.replace("MLP Small", "MLP, 1 layer").replace("MLP Deep", "MLP, 3 layers")
    return s


master = pd.read_csv(RUN / "tables" / "MASTER_test_all_systems_full.csv")
# The study's 45 systems: the pipeline's 40 plus Qwen3.8-27B at both thresholds
# and the three Claude models, scored by analysis/combined_significance.py.
_new = pd.read_csv(RUN / "outputs" / "prompt_llm" / "claude" / "qwen38_claude_test_metrics.csv")
_new = _new.rename(columns={"ci_low": "wf1_lo", "ci_high": "wf1_hi"}).assign(Tier="Prompt-based")
_new["Model"] = _new.System
master = pd.concat([master, _new[["System", "Model", "Tier", "weighted_f1", "biomarker_f1",
                                  "errors", "wf1_lo", "wf1_hi"]]], ignore_index=True)
assert len(master) == 45, len(master)
master = master.sort_values(["weighted_f1", "errors"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Figure 1: ranked systems with bootstrap intervals
# --------------------------------------------------------------------------
def fig_ranking():
    d = master[master.weighted_f1 >= 0.95].reset_index(drop=True)
    lead_lo, lead_hi = master.loc[0, "wf1_lo"], master.loc[0, "wf1_hi"]
    n_overlap = int((master.wf1_hi >= lead_lo).sum())
    n = len(d)
    y = np.arange(n)[::-1]

    fig, ax = plt.subplots(figsize=(6.3, 8.3))
    ax.axvspan(lead_lo, lead_hi, color="#efefef", zorder=0, lw=0)
    for i, r in d.iterrows():
        t = TIERS[r.Tier]
        ax.plot([r.wf1_lo, r.wf1_hi], [y[i]] * 2, color=t["color"], lw=1.3,
                solid_capstyle="butt", zorder=2)
        ax.plot(r.weighted_f1, y[i], marker=t["marker"], ms=t["ms"], color=t["color"],
                mec="white", mew=0.6, zorder=3, ls="none")

    # Where the overlap with the leader's interval ends.
    cut = y[n_overlap - 1] - 0.5
    ax.plot([0.92, 1.0135], [cut, cut], color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1,
            clip_on=False)
    ax.text(1.0045, cut + 0.35, f"{n_overlap} systems\nabove this line\noverlap the\nleader's interval\n(shaded)",
            fontsize=7.4, color=MUTED, va="bottom", ha="left", linespacing=1.25)

    ax.set_yticks(y)
    ax.set_yticklabels([tidy(s) for s in d.System])
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(0.92, 1.0025)
    ax.set_xticks([0.92, 0.94, 0.96, 0.98, 1.00])
    ax.set_xticklabels(["0.92", "0.94", "0.96", "0.98", "1.00"])
    ax.set_xlabel("Weighted F1 on the held-out test set (95% bootstrap interval)", labelpad=6)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.spines["left"].set_visible(False)

    handles = [Line2D([], [], color=t["color"], marker=t["marker"], ms=t["ms"], mec="white",
                      mew=0.6, lw=1.3, label=t["label"]) for t in TIERS.values()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.005), ncol=3,
              frameon=False, handlelength=1.8, columnspacing=1.4, handletextpad=0.5,
              borderaxespad=0.2)
    save(fig, "ranking")


# --------------------------------------------------------------------------
# Figure 2: recall by modality
# --------------------------------------------------------------------------
def fig_modality():
    m = pd.read_csv(MODALITY_CSV)
    m = m[m.biomarker_records >= 5].sort_values("mean_recall").reset_index(drop=True)
    names = {"Imaging (PET)": "PET", "Imaging (MRI/MRS)": "MRI / MRS",
             "Fluid (exosomal)": "Fluid, exosomal", "Microbiome / stool": "Microbiome / stool"}
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    y = np.arange(len(m))
    ax.axvline(0.90, color=MUTED, lw=0.7, ls=(0, (1.5, 2)), zorder=1)
    ax.text(1.07, len(m) - 0.45, "Mean", ha="left", va="bottom", fontsize=7.6, color=MUTED)
    for i, r in m.iterrows():
        weak = r.mean_recall < 0.90
        col = "#B8420B" if weak else "#3b3b3b"
        ax.plot([r.min_recall, r.max_recall], [i, i], color="#9a9a9a", lw=1.6,
                solid_capstyle="round", zorder=2)
        ax.plot(r.mean_recall, i, "o", ms=6.5, mfc="white" if not weak else col,
                mec=col, mew=1.4, zorder=3)
        # Value label sits left of the range so it never meets the point.
        ax.text(1.07, i, f"{r.mean_recall:.2f}", ha="left", va="center",
                fontsize=8, color=col, fontweight="bold" if weak else "normal")
    labels = [f"{names.get(s, s)}  (n = {int(k)})" for s, k in zip(m.Modality, m.biomarker_records)]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.6, len(m) - 0.2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "0.9", "1.0"])
    ax.set_xlabel("Recall on biomarker outcomes (out of fold, 8 encoders)", labelpad=6)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    dig = m[m.Modality == "Digital"].iloc[0]
    ax.text(dig.max_recall + 0.025, 0, f"{dig.pct_of_all_false_negatives:.1f}% of all misses",
            va="center", ha="left", fontsize=7.8, color="#B8420B")
    save(fig, "modality_recall")


# --------------------------------------------------------------------------
# Supplementary Figure S1: significance matrix (lower triangle)
# --------------------------------------------------------------------------
def fig_significance():
    sig = pd.read_csv(RUN / "outputs" / "final_results_45" / "significance_pairwise_test_45.csv")
    sig["A"] = sig.system_A.map(display_name)
    sig["B"] = sig.system_B.map(display_name)
    order_raw = master.System.tolist()
    key = {display_name(s): s for s in order_raw}
    # Match master names to the significance file's names.
    names = [display_name(s) for s in order_raw]
    known = set(sig.A) | set(sig.B)
    missing = [s for s in names if s not in known]
    assert not missing, f"names not matched: {missing}"
    idx = {s: i for i, s in enumerate(names)}
    n = len(names)
    M = np.full((n, n), np.nan)
    for r in sig.itertuples():
        i, j = idx[r.A], idx[r.B]
        lo, hi = max(i, j), min(i, j)
        M[lo, hi] = 1.0 if r.sig_holm else 0.0
    assert np.isfinite(M[np.tril_indices(n, -1)]).all()

    fig, ax = plt.subplots(figsize=(7.2, 7.4))
    cmap = matplotlib.colors.ListedColormap(["#e9e9e9", "#2c5d8a"])
    Mask = np.ma.masked_invalid(M)
    ax.pcolormesh(np.arange(n + 1), np.arange(n + 1), Mask, cmap=cmap, vmin=0, vmax=1,
                  edgecolors="white", linewidth=0.5)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(n, 1)
    ax.set_aspect("equal")
    lab = [tidy(s) for s in names]
    ax.set_yticks(np.arange(1, n) + 0.5)
    ax.set_yticklabels(lab[1:], fontsize=6.6)
    ax.set_xticks(np.arange(n - 1) + 0.5)
    ax.set_xticklabels(lab[:-1], fontsize=6.6, rotation=90)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    # The leading group: every system with five or fewer errors.
    k = int((master.errors <= 5).sum())
    assert (master.errors.iloc[:k] <= 5).all(), "leading group is not a prefix of the order"
    within = [r for r in sig.itertuples() if idx[r.A] < k and idx[r.B] < k]
    assert len(within) == k * (k - 1) // 2 and not any(r.sig_holm for r in within)
    words = {12: "Twelve", 13: "Thirteen", 14: "Fourteen", 15: "Fifteen", 16: "Sixteen"}
    ax.add_patch(Rectangle((0, 1), k - 1, k - 1, fill=False, ec="#C0392B", lw=1.2))
    ax.text(k - 0.4, k / 2 + 0.5, f"{words.get(k, k)} strongest systems:\nno pair differs after\nHolm correction",
            fontsize=8, color="#C0392B", va="center", ha="left", linespacing=1.3)
    handles = [Patch(fc="#2c5d8a", label="Significant after Holm correction (the column system is better)"),
               Patch(fc="#e9e9e9", label="Not significantly different")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.99), frameon=False,
              fontsize=7.8, handlelength=1.2)
    save(fig, "significance")


# --------------------------------------------------------------------------
# Supplementary Figure S2: confusion matrices
# --------------------------------------------------------------------------
def fig_confusion():
    ds = pd.read_csv(RUN / "outputs" / "dataset" / "labeled_dataset.csv")
    oof_cl = pd.read_csv(RUN / "outputs" / "classical_ml" / "oof_predictions.csv")
    oof_be = pd.read_csv(RUN / "outputs" / "bert_models" / "oof_predictions_merged.csv")
    te_cl = pd.read_csv(RUN / "outputs" / "classical_ml" / "test_predictions.csv")
    te_be = pd.read_csv(RUN / "outputs" / "bert_models" / "test_predictions_merged.csv")
    te_pr = pd.read_csv(RUN / "outputs" / "prompt_llm" / "test_predictions.csv")
    te_q38 = pd.read_csv(RUN / "outputs" / "prompt_llm" / "qwen38" / "test_predictions.csv")
    assert (te_q38.y_true.values == te_pr.y_true.values).all()
    panels = [("TF-IDF only", "Logistic regression", "Classical | Logistic Regression", oof_cl, te_cl),
              ("TF-IDF + FastText", "One-layer MLP", "FastText | MLP Small", oof_cl, te_cl),
              ("TF-IDF + BioWordVec", "SVM", "BioWordVec | SVM", oof_cl, te_cl),
              ("Fine-tuned encoder", "BioLinkBERT", "BioLinkBERT", oof_be, te_be),
              ("Prompted, tuned threshold", "Qwen2.5-7B", "Qwen2.5-7B-Instruct (tuned threshold)", None, te_pr),
              ("Prompted, tuned threshold", "Qwen3.8-27B", "Qwen3.8-27B (tuned threshold)", None, te_q38)]
    cmap = plt.get_cmap("Blues")

    def head(ax, tier, model):
        # model name in bold so the reader can tell the columns apart; group underneath
        ax.text(0, 1.20, model, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
        ax.text(0, 1.05, tier, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8, color=MUTED)

    fig, axes = plt.subplots(2, 6, figsize=(10.0, 4.4), gridspec_kw=dict(wspace=0.16, hspace=0.22))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.86, bottom=0.12)
    for c, (tier, model, col, oof, te) in enumerate(panels):
        for r, src in enumerate([oof, te]):
            ax = axes[r, c]
            if src is None:
                ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5); ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.text(0.5, 0.5, "No out-of-fold panel:\nprompted models\nare not trained",
                        ha="center", va="center", fontsize=7.8, color=MUTED, transform=ax.transAxes)
                if r == 0:
                    head(ax, tier, model)
                continue
            y, p = src.y_true.values, src[col].values
            cm = np.array([[((y == a) & (p == b)).sum() for b in (0, 1)] for a in (0, 1)])
            rowpct = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1.25)
            for a in range(2):
                for b in range(2):
                    dark = rowpct[a, b] > 0.6
                    ax.text(b, a - 0.1, f"{cm[a, b]:,}", ha="center", va="center", fontsize=9.5,
                            fontweight="bold", color="white" if dark else INK)
                    ax.text(b, a + 0.2, f"{100*rowpct[a, b]:.1f}%", ha="center", va="center",
                            fontsize=7.4, color="white" if dark else MUTED)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Not biomarker", "Biomarker"] if r == 1 else ["", ""], fontsize=7.4)
            ax.set_yticklabels(["Not biomarker", "Biomarker"] if c == 0 else ["", ""], fontsize=7.4)
            ax.tick_params(length=0, pad=3)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                head(ax, tier, model)
            if r == 1 and c == 0:
                pass
    for r, txt in enumerate(["Out of fold,\ntraining set\n(n = 1,559)", "Held-out\ntest set\n(n = 341)"]):
        pos = axes[r, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, txt, ha="left", va="center", fontsize=8.2,
                 linespacing=1.3)
    fig.text(0.565, 0.015, "Predicted label  (rows are the annotated label; percentages are per row)",
             ha="center", fontsize=8.2)
    save(fig, "confusion")



# --------------------------------------------------------------------------
# Supplementary Figure S4: the three Claude models
# --------------------------------------------------------------------------
def fig_claude_confusion():
    """Claude was never trained here, so its training-set replies are as fair a
    test as its test-set replies, and 1,559 records show the pattern better.
    A reply that did not open with Yes or No gets its own column rather than
    being folded into a miss or a false alarm."""
    raw = pd.read_csv(RUN / "outputs" / "prompt_llm" / "claude" / "raw_replies.csv")
    ds = pd.read_csv(RUN / "outputs" / "dataset" / "labeled_dataset.csv")
    tr = np.load(RUN / "outputs" / "dataset" / "train_idx.npy")
    te = np.load(RUN / "outputs" / "dataset" / "test_idx.npy")
    y = ds.has_biomarker.values
    models = [("claude-opus-5", "Claude Opus 5"), ("claude-sonnet-5", "Claude Sonnet 5"),
              ("claude-haiku-4-5", "Claude Haiku 4.5")]
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.6), gridspec_kw=dict(wspace=0.14, hspace=0.30))
    fig.subplots_adjust(left=0.15, right=0.99, top=0.86, bottom=0.13)
    for c, (mid, name) in enumerate(models):
        g = raw[(raw.model == mid) & (raw.run == "run1")].set_index("row").pred
        for r, idx in enumerate([tr, te]):
            ax = axes[r, c]
            p, yy = g.reindex(idx).values, y[idx]
            assert not np.isnan(p).any(), f"missing replies for {mid}"
            cm = np.array([[((yy == a) & (p == b)).sum() for b in (0, 1, -1)] for a in (0, 1)])
            rowpct = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(rowpct, cmap=cmap, vmin=0, vmax=1.25, aspect="auto")
            for a in range(2):
                for b in range(3):
                    dark = rowpct[a, b] > 0.6
                    ax.text(b, a - 0.12, f"{cm[a, b]:,}", ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white" if dark else INK)
                    ax.text(b, a + 0.2, f"{100*rowpct[a, b]:.1f}%", ha="center", va="center",
                            fontsize=7, color="white" if dark else MUTED)
            ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Not\nbiomarker", "Biomarker", "Unreadable"] if r == 1 else ["", "", ""],
                               fontsize=7.4)
            ax.set_yticklabels(["Not biomarker", "Biomarker"] if c == 0 else ["", ""], fontsize=7.4)
            ax.tick_params(length=0, pad=3)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.text(0, 1.20, name, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9.5, fontweight="bold", color=INK)
                ax.text(0, 1.05, "Prompted, first word scored", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=8, color=MUTED)
    for r, txt in enumerate(["Training set\n(n = 1,559;\nnever trained on)", "Held-out\ntest set\n(n = 341)"]):
        pos = axes[r, 0].get_position()
        fig.text(0.012, (pos.y0 + pos.y1) / 2, txt, ha="left", va="center", fontsize=8.2,
                 linespacing=1.3)
    fig.text(0.57, 0.012, "Reply  (rows are the annotated label; percentages are per row)",
             ha="center", fontsize=8.2)
    save(fig, "claude_confusion")

# --------------------------------------------------------------------------
# Supplementary Figure S3: corpus overview
# --------------------------------------------------------------------------
def fig_corpus():
    ds = pd.read_csv(RUN / "outputs" / "dataset" / "labeled_dataset.csv")
    comp = pd.read_csv(RUN / "tables" / "table_modality_composition.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), gridspec_kw=dict(hspace=0.55, wspace=0.42))

    ax = axes[0, 0]
    k = ds.has_biomarker.value_counts().reindex([0, 1])
    bars = ax.bar(["Not biomarker", "Biomarker"], k.values, color=["#9fb7cc", "#D55E00"],
                  width=0.6, edgecolor="none")
    for b, v in zip(bars, k.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 25, f"{v:,}  ({100*v/len(ds):.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1550)
    ax.set_ylabel("Outcome measures")
    ax.set_title("(a) Class balance", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

    ax = axes[0, 1]
    c = comp[comp.Biomarker > 0].copy()
    rename = {"Imaging --- MRI/MRS": "MRI / MRS", "Imaging --- PET": "PET",
              "Fluid --- exosomal": "Fluid, exosomal", "Imaging --- retinal (OCT)": "Retinal (OCT)",
              "Pharmacokinetic / immunogenicity": "Pharmacokinetic", "Clinical / behavioural": "Clinical"}
    c["name"] = c.Modality.map(lambda s: rename.get(s, s))
    c = c.sort_values("Biomarker")
    ax.barh(c.name, c.Biomarker, color="#009E73", height=0.62, edgecolor="none")
    for yy, v in enumerate(c.Biomarker):
        ax.text(v * 1.12, yy, f"{v}", va="center", fontsize=7.6, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 900)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.tick_params(axis="y", length=0, labelsize=7.6)
    ax.set_xlabel("Biomarker outcome measures (log scale)")
    ax.set_title("(b) Biomarker outcomes by modality", loc="left")
    ax.spines["left"].set_visible(False)

    ax = axes[1, 0]
    per = ds.groupby("NCTId").size()
    ax.hist(per, bins=np.arange(0.5, per.max() + 2, 2), color="#56B4E9", edgecolor="white", lw=0.5)
    ax.set_xlabel("Outcome measures per trial")
    ax.set_ylabel("Trials")
    ax.set_title("(c) Outcome measures per trial", loc="left")
    ax.text(0.97, 0.93, f"median {int(per.median())}, max {per.max()}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.8, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

    ax = axes[1, 1]
    texts = ds.clean_freeform.fillna("").tolist()
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
        lens = np.array([len(tok(t, add_special_tokens=True)["input_ids"]) for t in texts])
        xlabel, limit_label = "Tokens per outcome measure", "128-token limit"
    except Exception as e:  # tokenizer unavailable: fall back to words
        print("tokenizer unavailable, plotting words:", e)
        lens = np.array([len(t.split()) for t in texts])
        xlabel, limit_label = "Words per outcome measure", "128 words"
    over = int((lens > 128).sum())
    ax.hist(np.clip(lens, 0, 300), bins=np.arange(0, 305, 8), color="#E69F00", edgecolor="white", lw=0.5)
    ax.axvline(128, color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.text(133, ax.get_ylim()[1] * 0.93, f"{limit_label}\n{over:,} records ({100*over/len(lens):.1f}%)\nare truncated",
            fontsize=7.6, va="top", ha="left", color=INK, linespacing=1.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Outcome measures")
    ax.set_title("(d) Length of the model input", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "corpus_overview")
    return over, len(lens), float(np.median(lens))


if __name__ == "__main__":
    fig_ranking()
    fig_modality()
    fig_significance()
    fig_confusion()
    fig_claude_confusion()
    print("corpus:", fig_corpus())
    if os.environ.get("BMO_SUBMISSION_DIR"):
        sub = Path(os.environ["BMO_SUBMISSION_DIR"])
        shutil.copy(OUT / "ranking.pdf", sub / "Figure1.pdf")
        shutil.copy(OUT / "modality_recall.pdf", sub / "Figure2.pdf")
        print(f"copied Figure1.pdf and Figure2.pdf to {sub}")
