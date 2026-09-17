#%% md
# 09 — Manuscript figures, tables, and the run manifest

Assembles the publication artefacts from the result files, then seals the run
with output checksums.

Every figure is written as both **PDF** (vector, for the manuscript) and **PNG**
(for the repository README and slides). Figure design follows the rules fixed in
notebook 00: colour-blind-safe palette, information never carried by hue alone,
and verified greyscale separation so the figures survive black-and-white
printing.
#%% code
import os, sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

master = pd.read_csv(C.OUT_FINAL / "MASTER_test_all_systems.csv")
df, train_idx, test_idx = C.load_data()
y_test = df["has_biomarker"].values[test_idx]
print(f"{len(master)} systems | test n={len(y_test)}")
#%% md
## Table 1 — main results, manuscript format

Rounded to the precision the manuscript actually reports. Publishing four
decimal places on a 341-record test set implies a resolution the data does not
have; three is already generous, since one record is worth ~0.003 weighted F1.
#%% code
t1 = master.copy()
for c in ("weighted_f1", "biomarker_f1", "macro_f1", "auc", "precision",
          "recall", "specificity", "mcc"):
    if c in t1.columns:
        t1[c] = t1[c].round(3)
COLS = [c for c in ["System", "Tier", "weighted_f1", "biomarker_f1", "auc",
                    "precision", "recall", "TN", "FP", "FN", "TP", "errors"]
        if c in t1.columns]
table1 = t1[COLS]
C.save_table(table1, C.TABLES / "table1_main_results.csv")

def to_latex(d, path, caption, label):
    body = d.to_latex(index=False, escape=True, longtable=False,
                      column_format="l" * 2 + "r" * (len(d.columns) - 2))
    path.write_text(f"\\begin{{table}}[htbp]\n\\centering\n\\small\n"
                    f"\\caption{{{caption}}}\n\\label{{{label}}}\n{body}\\end{{table}}\n")
    print(f"latex  -> {path}")

to_latex(table1, C.TABLES / "table1_main_results.tex",
         "Performance of all systems on the held-out test set "
         f"($n={len(y_test)}$ outcome measures from trials not seen in training). "
         "Tiers: Classical (TF-IDF), FastText, BioWordVec, Transformer, Prompt-based.",
         "tab:main_results")
print(table1.head(15).to_string(index=False))
#%% md
## Table 2 — the representation experiment

Identical estimators and folds; only the text representation differs. This is
the table that carries the paper's central argument.
#%% code
p = C.OUT_CLASSICAL / "representation_effect.csv"
if p.exists():
    rep = pd.read_csv(p).round(3)
    C.save_table(rep, C.TABLES / "table2_representation_effect.csv")
    to_latex(rep, C.TABLES / "table2_representation_effect.tex",
             "Effect of text representation, holding estimator, folds and "
             "hyperparameter grid constant. Values are test-set weighted F1.",
             "tab:representation")
    print(rep.to_string(index=False))
else:
    print("run notebook 02 first")
#%% md
## Table 3 — overfitting audit and tuning budget

The evidence behind the manuscript's claim about generalisation. Reporting the
number of configurations searched next to the train-validation gap is what
turns "we tuned carefully" into a checkable statement.
#%% code
parts = []
for p, tier in ((C.OUT_BERT / "overfitting_audit_merged.csv", "Transformer"),
                (C.OUT_CLASSICAL / "overfitting_audit.csv", "Classical tiers")):
    if p.exists():
        d = pd.read_csv(p); d["Group"] = tier; parts.append(d)
if parts:
    audit = pd.concat(parts, ignore_index=True).sort_values("gap", ascending=False)
    audit[["train_f1", "val_f1", "gap"]] = audit[["train_f1", "val_f1", "gap"]].round(4)
    C.save_table(audit, C.TABLES / "table3_overfitting_audit.csv")
    to_latex(audit, C.TABLES / "table3_overfitting_audit.tex",
             "Train-to-validation weighted F1 gap for every supervised system, "
             "with the number of hyperparameter configurations evaluated. "
             "Status: OK $<0.04$, WATCH $<0.10$, OVERFIT $\\geq0.10$.",
             "tab:overfitting")
    print(audit.to_string(index=False))
    bad = audit[audit["status"] != "OK"]
    print(f"\n{len(bad)} system(s) exceed the 0.04 threshold and must be "
          "discussed explicitly in the manuscript."
          if len(bad) else "\nAll systems within the 0.04 threshold.")
#%% md
## Figure: representation versus classifier

The paper's central comparison, drawn as a dumbbell plot. Each row is one
classifier; the line runs from its score under TF-IDF alone to its score with
biomedical embeddings, with the general-domain embedding marked in between.

This form is chosen because the quantity of interest is a *change within* each
classifier, not a comparison between classifiers. Grouped bars would invite the
reader to compare across rows, which is not the experiment: the design holds the
classifier fixed and varies only the representation.
#%% code
import matplotlib.pyplot as plt

rep = pd.read_csv(C.OUT_CLASSICAL / "representation_effect.csv")
NAME = {"Logistic Regression": "Logistic regression", "SVM": "Support vector machine",
        "Random Forest": "Random forest", "XGBoost": "XGBoost",
        "Decision Tree": "Decision tree", "MLP Small": "Neural network (1 layer)",
        "MLP Deep": "Neural network (3 layers)"}
rep = rep.sort_values("BioWordVec")
y = np.arange(len(rep))

fig, ax = plt.subplots(figsize=(6.4, 3.1))
for i, (_, r) in enumerate(rep.iterrows()):
    lo, hi = min(r.Classical, r.BioWordVec), max(r.Classical, r.BioWordVec)
    ax.plot([lo, hi], [y[i], y[i]], color="0.72", lw=2.2, solid_capstyle="round", zorder=1)
    ax.plot(r.Classical,  y[i], marker="o", ms=5.5, color="#0072B2",
            markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    ax.plot(r.FastText,   y[i], marker="s", ms=4.6, color="#56B4E9",
            markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    ax.plot(r.BioWordVec, y[i], marker="^", ms=5.8, color="#009E73",
            markeredgecolor="black", markeredgewidth=0.4, zorder=3)
    d = r.BioWordVec - r.Classical
    ax.text(1.001, y[i], f"{d:+.3f}", va="center", fontsize=6.6,
            color="#00694E" if d > 0 else "#8A3324")

ax.set_yticks(y, [NAME.get(m, m) for m in rep["Model"]], fontsize=7.5)
ax.set_xlabel("Held-out weighted F1")
ax.set_title("Effect of text representation, classifier held fixed", loc="left")
ax.set_xlim(0.87, 1.0)
ax.grid(axis="x", alpha=0.3, lw=0.4); ax.grid(axis="y", visible=False)
ax.set_axisbelow(True)
ax.text(1.001, len(rep) - 0.35, "change", fontsize=6.6, color="0.3", va="center")
handles = [plt.Line2D([], [], marker=mk, ls="", color=c, ms=5.4,
                      markeredgecolor="black", markeredgewidth=0.4, label=lab)
           for mk, c, lab in (("o", "#0072B2", "TF-IDF"),
                              ("s", "#56B4E9", "+ FastText-300 (general)"),
                              ("^", "#009E73", "+ BioWordVec-200 (biomedical)"))]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
for ext in ("pdf", "png"):
    fig.savefig(C.FIGS / f"fig_representation_effect.{ext}", bbox_inches="tight")
plt.close(fig)
print("figure ->", C.FIGS / "fig_representation_effect.pdf")
print(f"BioWordVec improves {int((rep.BioWordVec > rep.Classical).sum())}/7 classifiers, "
      f"mean {(rep.BioWordVec - rep.Classical).mean():+.4f}")
print(f"FastText improves   {int((rep.FastText > rep.Classical).sum())}/7 classifiers, "
      f"mean {(rep.FastText - rep.Classical).mean():+.4f}")

#%% md
## Figure: confusion matrices, out-of-fold and held-out

Two rows for each tier leader. The upper row uses **out-of-fold predictions on
the training partition** (1,559 records, each predicted by a model that never
saw it); the lower uses the held-out test partition (341 records).

Both are worth showing. The test partition is the headline evaluation but
contains so few errors -- one to six for the leading systems -- that its
confusion matrix conveys little about the error *pattern*. The out-of-fold
matrix has four to five times as many errors and shows the asymmetry between
missed biomarkers and false alarms clearly. Neither involves leakage: an
out-of-fold prediction comes from a fold in which that record was held out.
#%% code
import matplotlib.pyplot as plt

# Prediction files key classical systems as "Tier | Model" while the master
# table displays them as "Tier + Model". Matching on the display string alone
# silently dropped three of the five tier leaders from an earlier version of
# this figure, so the key is rebuilt from Tier and Model.
full_path = C.TABLES / "MASTER_test_all_systems_full.csv"
full = pd.read_csv(full_path) if full_path.exists() else master
key_of = {r["System"]: (f"{r['Tier']} | {r['Model']}"
                        if r["Tier"] in ("Classical", "FastText", "BioWordVec")
                        else r["Model"])
          for _, r in full.iterrows()} if "Model" in full.columns else {}

test_pred, oof_pred = {}, {}
for path, sink in ((C.OUT_CLASSICAL / "test_predictions.csv", test_pred),
                   (C.OUT_BERT / "test_predictions_merged.csv", test_pred),
                   (C.OUT_LLM / "test_predictions.csv", test_pred),
                   (C.OUT_CLASSICAL / "oof_predictions.csv", oof_pred),
                   (C.OUT_BERT / "oof_predictions_merged.csv", oof_pred)):
    if path.exists():
        d = pd.read_csv(path)
        for c in d.columns:
            if c != "y_true":
                sink[c] = d[c].values

y_train = df["has_biomarker"].values[train_idx]

leaders = []
for tier in ("Classical", "FastText", "BioWordVec", "Transformer", "Prompt-based"):
    sub = master[master["Tier"] == tier].sort_values("weighted_f1", ascending=False)
    if not len(sub):
        continue
    name = sub.iloc[0]["System"]
    leaders.append((tier, name, key_of.get(name, name)))
print("tier leaders:", [(t, k) for t, _, k in leaders])

missing = [k for _, _, k in leaders if k not in test_pred]
assert not missing, f"no test predictions for {missing}; check the key mapping"

fig, axes = plt.subplots(2, len(leaders), figsize=(2.05 * len(leaders), 4.6))
for col, (tier, name, key) in enumerate(leaders):
    short = name if len(name) <= 26 else name[:24] + ".."
    if key in oof_pred:
        C.draw_confusion(axes[0, col], y_train, oof_pred[key],
                         title=f"{tier}\n{short}")
    else:
        axes[0, col].axis("off")
        axes[0, col].set_title(f"{tier}\n{short}\n(no out-of-fold predictions)",
                               loc="left", fontsize=7)
    C.draw_confusion(axes[1, col], y_test, test_pred[key])
    axes[1, col].set_xlabel("Predicted", fontsize=7.5)
    if col:
        axes[0, col].set_ylabel(""); axes[1, col].set_ylabel("")
axes[0, 0].set_ylabel("Out-of-fold\n(n=%d)\n\nAnnotated" % len(y_train), fontsize=7.5)
axes[1, 0].set_ylabel("Held-out test\n(n=%d)\n\nAnnotated" % len(y_test), fontsize=7.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(C.FIGS / f"fig_confusion_tier_leaders.{ext}", bbox_inches="tight")
plt.close(fig)
print("figure ->", C.FIGS / "fig_confusion_tier_leaders.pdf")

#%% md
## Figure — greyscale proof

Renders the palette as it will appear in colour and as a printer will render
it. Included in the repository as evidence the accessibility constraint was
tested rather than assumed.
#%% code
def luma(hexc):
    r, g, b = (int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b

fig, ax = plt.subplots(1, 2, figsize=(6.4, 1.9))
for k, t in enumerate(C.TIERS):
    st = C.TIER_STYLE[t]
    ax[0].barh(k, 1, color=st["color"], hatch=st["hatch"],
               edgecolor="black", linewidth=0.6)
    g = luma(st["color"])
    ax[1].barh(k, 1, color=(g, g, g), hatch=st["hatch"],
               edgecolor="black", linewidth=0.6)
for a, title in zip(ax, ("As displayed", "As printed (greyscale)")):
    a.set_yticks(range(len(C.TIERS)), C.TIERS, fontsize=7)
    a.set_xticks([]); a.set_title(title, fontsize=8, loc="left"); a.grid(False)
fig.tight_layout(); fig.savefig(C.FIGS / "fig_palette_greyscale_check.png")
plt.close(fig)
vals = sorted(luma(C.TIER_STYLE[t]["color"]) for t in C.TIERS)
print(f"minimum greyscale separation between tiers: "
      f"{min(b-a for a, b in zip(vals, vals[1:])):.3f}")
print("every tier additionally carries a unique hatch pattern, so separation "
      "does not depend on greyscale value alone")
#%% md
## Seal the run — output checksums

The manifest from notebook 00 gains a checksum for every result file produced.
Any figure or table in the manuscript can then be traced to a file whose
contents are pinned.
#%% code
manifest = json.loads((C.PROV / "run_manifest.json").read_text())
outputs = {}
for d in (C.OUT_FINAL, C.TABLES, C.OUT_BERT, C.OUT_CLASSICAL, C.OUT_LLM):
    for f in sorted(d.glob("*.csv")):
        outputs[str(f.relative_to(RUN_DIR))] = C.sha256(f)
for f in sorted(C.FIGS.glob("*.pdf")):
    outputs[str(f.relative_to(RUN_DIR))] = C.sha256(f)

manifest["outputs_sha256"] = outputs
manifest["n_systems"] = int(len(master))
manifest["best_system"] = {
    "name": str(master.iloc[0]["System"]),
    "weighted_f1": float(master.iloc[0]["weighted_f1"]),
    "errors": int(master.iloc[0]["errors"]),
}
(C.PROV / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
(C.PROV / "CHECKSUMS.txt").write_text(
    "\n".join(f"{h}  {p}" for p, h in sorted(outputs.items())) + "\n")

print(f"{len(outputs)} output files checksummed")
print(f"best system: {manifest['best_system']['name']} "
      f"(weighted F1 {manifest['best_system']['weighted_f1']:.4f}, "
      f"{manifest['best_system']['errors']} errors)")
print("manifest ->", C.PROV / "run_manifest.json")
#%% md
## Manuscript checklist

Generated from the results rather than from memory, so the numbers quoted in
the text can be checked against it directly.
#%% code
lines = [
    "MANUSCRIPT NUMBERS — generated from this run",
    "=" * 60,
    f"corpus: {manifest['dataset']['n_records']:,} outcome measures from "
    f"{manifest['dataset']['n_trials']:,} trials",
    f"positive rate: {manifest['dataset']['positive_rate']*100:.2f}%",
    f"train / test: {manifest['dataset']['n_train']:,} / {manifest['dataset']['n_test']:,}",
    f"systems evaluated: {len(master)}",
    "",
    f"best overall: {master.iloc[0]['System']} — weighted F1 "
    f"{master.iloc[0]['weighted_f1']:.4f}, {int(master.iloc[0]['errors'])} errors",
]
for tier in ("Classical", "FastText", "BioWordVec", "Transformer", "Prompt-based"):
    sub = master[master["Tier"] == tier].sort_values("weighted_f1", ascending=False)
    if len(sub):
        lines.append(f"best {tier:13s}: {sub.iloc[0]['System']:32s} "
                     f"F1 {sub.iloc[0]['weighted_f1']:.4f}  "
                     f"{int(sub.iloc[0]['errors'])} errors")

sigp = C.OUT_FINAL / "significance_pairwise_test.csv"
if sigp.exists():
    s = pd.read_csv(sigp)
    lines += ["", f"pairwise comparisons: {len(s)}",
              f"  significant uncorrected: {int(s['sig_raw'].sum())} "
              f"(~{0.05*len(s):.0f} expected by chance)",
              f"  significant Holm:        {int(s['sig_holm'].sum())}",
              f"  significant BH-FDR:      {int(s['sig_fdr'].sum())}",
              f"  pairs with identical predictions: {int((s['discordant']==0).sum())}"]
lines += ["", "REMINDER: one test-set error = "
          f"{1/len(y_test):.4f} accuracy. Differences of one or two errors are "
          "not meaningful; quote the corrected significance tests, not the ranking."]
text = "\n".join(lines)
(C.TABLES / "manuscript_numbers.txt").write_text(text + "\n")
print(text)
#%% md
## Done

The run is complete and sealed. `outputs/provenance/run_manifest.json` records
the environment, seeds, input checksums and output checksums; `CHECKSUMS.txt`
lists every result file.
