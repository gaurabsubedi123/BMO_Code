#%% md
# 01 — Corpus, labels, and descriptive statistics

Everything the Methods and Results sections need to describe the data.

## A deliberate choice: the split is reused, not regenerated

This notebook **loads** `train_idx.npy` / `test_idx.npy` rather than recreating
them. That is intentional and important.

Regenerating the split — even with the same seed — risks a different partition
if any library changes its shuffling internals, and every number in this re-run
would then be incomparable with the earlier study. Reusing the saved indices
means the new models are evaluated on exactly the records the previous models
were, so the comparison across runs is meaningful rather than approximate.

The split's integrity is re-verified here rather than assumed.
#%% code
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd

RUN_DIR = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))
sys.path.insert(0, str(RUN_DIR))
import bmo_common as C

df, train_idx, test_idx = C.load_data()
print(f"records {len(df):,} | columns: {list(df.columns)}")
df.head(3)
#%% md
## Corpus composition
#%% code
n_pos = int(df["has_biomarker"].sum())
summary = {
    "Records (outcome measures)": f"{len(df):,}",
    "Trials (unique NCTId)":      f"{df['NCTId'].nunique():,}",
    "Biomarker outcomes":         f"{n_pos:,} ({n_pos/len(df)*100:.2f}%)",
    "Non-biomarker outcomes":     f"{len(df)-n_pos:,} ({(1-n_pos/len(df))*100:.2f}%)",
    "Outcomes per trial (median)": f"{df.groupby('NCTId').size().median():.0f}",
    "Outcomes per trial (max)":    f"{df.groupby('NCTId').size().max():,}",
    "Training records":            f"{len(train_idx):,}",
    "Test records":                f"{len(test_idx):,}",
}
for k, v in summary.items():
    print(f"{k:32s} {v}")
#%% md
### Why trial-level grouping is not optional

The distribution below is the argument. A handful of trials contribute dozens
of outcome measures each, and outcomes within one trial share phrasing,
abbreviations and formatting conventions. Splitting at the record level would
place near-duplicates of a training record into the test set and inflate every
reported metric.
#%% code
per_trial = df.groupby("NCTId").size()
print(per_trial.describe().round(2).to_string())
print(f"\ntop 5 trials contribute {per_trial.nlargest(5).sum()} records "
      f"({per_trial.nlargest(5).sum()/len(df)*100:.1f}% of the corpus)")

tr_trials, te_trials = set(df["NCTId"].values[train_idx]), set(df["NCTId"].values[test_idx])
print(f"\ntrials in train {len(tr_trials):,} | test {len(te_trials):,} | "
      f"shared {len(tr_trials & te_trials)}")
assert not (tr_trials & te_trials), "split is not trial-disjoint"
print("split verified: trial-disjoint")
#%% md
## Class balance across the split

The test set should mirror the training set's positive rate closely; a large
divergence would mean the held-out estimate is measuring a different problem.
#%% code
bal = pd.DataFrame([
    {"Partition": "Full corpus", "n": len(df),
     "positive": int(df["has_biomarker"].sum()),
     "positive_rate": df["has_biomarker"].mean()},
    {"Partition": "Train", "n": len(train_idx),
     "positive": int(df["has_biomarker"].values[train_idx].sum()),
     "positive_rate": df["has_biomarker"].values[train_idx].mean()},
    {"Partition": "Test", "n": len(test_idx),
     "positive": int(df["has_biomarker"].values[test_idx].sum()),
     "positive_rate": df["has_biomarker"].values[test_idx].mean()},
])
bal["positive_rate"] = bal["positive_rate"].round(4)
print(bal.to_string(index=False))
C.save_table(bal, C.TABLES / "class_balance.csv")
#%% md
## Text length

Justifies the 128-token sequence budget used for every transformer: if the
overwhelming majority of records fit well inside it, the budget is not the
binding constraint on performance and holding it fixed across models is fair.
#%% code
words = df["clean_freeform"].fillna("").str.split().str.len()
print(words.describe().round(1).to_string())
for q in (0.5, 0.9, 0.95, 0.99, 1.0):
    print(f"  {int(q*100):3d}th percentile: {words.quantile(q):6.0f} words")
print(f"\nrecords over 100 words: {(words > 100).sum()} ({(words>100).mean()*100:.2f}%)")
print("A 128-token budget covers the corpus; sequence length is not the "
      "limiting factor for any model.")
#%% md
## Biomarker modality composition

The binary label collapses a much finer annotation scheme: the `Annotation`
column carries 348 distinct categories. Recovering the modality behind each
positive record matters for two reasons — it describes what the corpus actually
contains, and it sets a hard limit on the multi-label extension proposed in the
Discussion.

**The mapping is not purely prefix-based, and it cannot be.** Most `UD_` entries
("unspecified matrix") are fluid analytes — CRP, IL-6, NfL, GFAP, cholesterol.
But `UD_GSH`, `UD_NAD`, `UD_NAD/NADH` and `UD_CK/ATPase` describe *in-vivo brain
spectroscopy* ("changes in brain GSH levels"), not blood assays, and prefix
matching alone files them under fluid. Those four, plus `MT_Myelin`, `CDM` and
`UD_AFD` (all MRI), `RETINAL_Aβ`/`OCT_Aβ` (retinal imaging) and `TMS_*`
(neurophysiology), are assigned from the outcome text.
#%% code
import re

MRS_UD = {"UD_GSH", "UD_NAD", "UD_NAD/NADH", "UD_CK/ATPase"}   # in-vivo spectroscopy
MRI_UD = {"UD_AFD", "MT_Myelin", "CDM"}

def modality(annotation):
    a = str(annotation).strip()
    first = a.upper().split(",")[0].strip()
    if first.startswith("ANALOG"):                    return "Clinical / behavioural"
    if a in MRS_UD or a in MRI_UD:                    return "Imaging --- MRI/MRS"
    if first.startswith(("RETINAL", "OCT")):          return "Imaging --- retinal (OCT)"
    if first.startswith("TMS"):                       return "Neurophysiology"
    if first.startswith("EXOSOME"):                   return "Fluid --- exosomal"
    if first.startswith("STOOL"):                     return "Microbiome / stool"
    if re.match(r"^(Aβ|AB|TAU|FDG|SV2A|TSPO)", first) or "PET" in first:
        return "Imaging --- PET"
    if any(k in first for k in ("MRI","ROI","NODDI","DTI","MRS","ASL","FMRI","WMH","VOLUM")):
        return "Imaging --- MRI/MRS"
    if first.startswith(("CSF","PLASMA","SERUM","SEURM","FLUID","BLOOD","URINE","SALIVA")):
        return "Fluid"
    if first.startswith("UD_"):                       return "Fluid"
    if any(k in first for k in ("EEG","PSG","MEG","ERP","EMG","ECG","EYE")):
        return "Neurophysiology"
    if first.startswith("DIGITAL") or "ACTIGRAPH" in first:  return "Digital"
    if any(k in first for k in ("APOE","GENOT","GENE","SNP","POLYGEN","RNA","DNA","OMIC")):
        return "Genetic / omics"
    if any(k in first for k in ("PK","ADA","ANTIBOD","IMMUNOGEN")):
        return "Pharmacokinetic / immunogenicity"
    # A single record is annotated simply "cognition". By content it is a
    # neuropsychological assessment -- every comparable outcome in the corpus is
    # ANALOG_neuropsychological evaluation and labelled non-biomarker -- yet it
    # carries has_biomarker = 1. It is classified here by what it measures, and
    # the resulting disagreement with the label is reported below rather than
    # hidden by the mapping.
    if first == "COGNITION":                          return "Clinical / behavioural"
    return "Unclassified"

df["modality"] = df["Annotation"].fillna("(none)").map(modality)

mod = (df.groupby("modality")
         .agg(Records=("has_biomarker", "size"),
              Biomarker=("has_biomarker", "sum"),
              Trials=("NCTId", "nunique"),
              Annotations=("Annotation", "nunique"))
         .reset_index()
         .rename(columns={"modality": "Modality"}))
mod["% of corpus"] = (mod["Records"] / len(df) * 100).round(1)
pos_total = int(df["has_biomarker"].sum())
mod["% of biomarkers"] = (mod["Biomarker"] / pos_total * 100).round(1)
mod = mod.sort_values("Records", ascending=False).reset_index(drop=True)
print(mod.to_string(index=False))

C.save_table(mod, C.TABLES / "table_modality_composition.csv")

unclassified = df[df["modality"] == "Unclassified"]
print(f"\nunclassified records: {len(unclassified)}")
if len(unclassified):
    print(unclassified["Annotation"].value_counts().to_string())
assert len(unclassified) == 0, (f"{len(unclassified)} records have no modality; "
                                "extend the mapping rather than dropping them")
print(f"all {len(df):,} records assigned; {pos_total} biomarker-positive")
#%% code
# LaTeX version for the manuscript.
# Two things to get right or the table silently breaks:
#   - "%" must be escaped, otherwise LaTeX treats the rest of the header line as
#     a comment and the tabular loses its columns;
#   - percentages must be formatted, or pandas emits 70.400000.
cols = ["Modality", "Records", "% of corpus", "Biomarker", "% of biomarkers",
        "Trials", "Annotations"]
tex = mod[cols].copy()
for c in ("% of corpus", "% of biomarkers"):
    tex[c] = tex[c].map(lambda v: f"{v:.1f}")
tex.columns = [c.replace("%", r"\%") for c in tex.columns]
body = tex.to_latex(index=False, escape=False,
                    column_format="l" + "r" * (len(cols) - 1))
(C.TABLES / "table_modality_composition.tex").write_text(
    "\\begin{table}[htbp]\n\\centering\n\\small\n"
    "\\caption{Composition of the annotated corpus by biomarker modality. "
    "Modality is recovered from the expert annotation label; the binary "
    "classification target collapses all non-\\texttt{ANALOG} categories into a "
    "single positive class. \\emph{Annotations} counts distinct fine-grained "
    "labels within each modality.}\n"
    "\\label{tab:modality}\n" + body + "\\end{table}\n")
print("latex ->", C.TABLES / "table_modality_composition.tex")

# The label is a deterministic function of the annotation prefix: worth
# asserting, because it is why no inter-annotator disagreement exists to measure.
conflict = df[((df["modality"] == "Clinical / behavioural") & (df["has_biomarker"] == 1)) |
              ((df["modality"] != "Clinical / behavioural") & (df["has_biomarker"] == 0))]
print(f"\nrecords where modality and binary label disagree: {len(conflict)} of {len(df):,}")
for _, r in conflict.iterrows():
    print(f"  {r['NCTId']} | annotation={r['Annotation']!r} | "
          f"modality={r['modality']} | has_biomarker={r['has_biomarker']}")
    print(f"    {str(r['clean_freeform'])[:150]}")
if len(conflict):
    print("\n-> These should be confirmed with the annotating group. They are too few "
          "to affect any metric, but they are the only direct evidence available on "
          "label reliability, since the corpus has a single annotator and therefore "
          "no inter-annotator agreement statistic.")
else:
    print("-> the binary label is exactly the ANALOG / non-ANALOG distinction")

# Feasibility of the multi-label extension proposed in the Discussion.
print("\nsupport for a multi-label extension (positive records per modality):")
for _, r in mod[mod["Biomarker"] > 0].iterrows():
    verdict = ("viable" if r["Biomarker"] >= 50 else
               "marginal" if r["Biomarker"] >= 20 else "too few to learn")
    print(f"  {r['Modality']:38s} {int(r['Biomarker']):4d}  {verdict}")

#%% md
## Figure — corpus overview

Four panels: class balance, the modality composition of the positive class,
outcomes per trial, and text length. Every panel is greyscale-safe — fills are
distinguished by hatch as well as tone, so nothing depends on colour surviving
black-and-white printing.

The modality panel uses a logarithmic count axis because the classes span three
orders of magnitude (1,337 clinical records down to a single
pharmacokinetic one). On a linear axis every biomarker modality would collapse
into an indistinguishable sliver against the clinical class, which is precisely
the imbalance the panel exists to show.
#%% code
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.2))

# (a) class balance. Two bars, each labelled directly beneath: hatching adds
# nothing a greyscale reader needs and only adds visual noise.
ax[0,0].bar(["No biomarker", "Biomarker"], [len(df)-n_pos, n_pos],
            color=["#0072B2", "#D55E00"], edgecolor="black", linewidth=0.6)
ax[0,0].set_ylabel("Outcome measures"); ax[0,0].set_title("(a) Class balance", loc="left")
for i, v in enumerate([len(df)-n_pos, n_pos]):
    ax[0,0].text(i, v, f"{v:,}\n{v/len(df)*100:.1f}%", ha="center", va="bottom", fontsize=7)
ax[0,0].set_ylim(0, max(len(df)-n_pos, n_pos) * 1.22)

# (b) modality composition of the positive class.
#
# One fill, no hatching. Every bar here belongs to the same category -- these
# are subdivisions of the biomarker class, not competing groups -- so varying
# the hatch would imply a distinction that does not exist and would clutter a
# panel whose message is simply the ordering and the magnitudes.
m = mod[mod["Biomarker"] > 0].sort_values("Biomarker")
ypos = np.arange(len(m))
LABEL = {"Imaging --- MRI/MRS": "MRI / MRS", "Imaging --- PET": "PET",
         "Imaging --- retinal (OCT)": "Retinal (OCT)",
         "Fluid --- exosomal": "Fluid (exosomal)",
         "Pharmacokinetic / immunogenicity": "Pharmacokinetic",
         "Clinical / behavioural": "Clinical (see text)"}
ax[0,1].barh(ypos, m["Biomarker"], color="#009E73", edgecolor="black", linewidth=0.5)
for i, v in enumerate(m["Biomarker"]):
    ax[0,1].text(v * 1.18, ypos[i], f"{int(v)}", va="center", fontsize=6.5)
ax[0,1].set_yticks(ypos, [LABEL.get(t, t) for t in m["Modality"]], fontsize=6.5)
ax[0,1].set_xscale("log"); ax[0,1].set_xlim(0.7, 900)
ax[0,1].xaxis.set_major_formatter(ScalarFormatter())
ax[0,1].set_xlabel("Biomarker outcome measures (log scale)")
ax[0,1].set_title("(b) Modality of the biomarker class", loc="left")
ax[0,1].grid(axis="y", visible=False)

# (c) outcomes per trial
ax[1,0].hist(per_trial, bins=30, color="#56B4E9", edgecolor="black", linewidth=0.4)
ax[1,0].set_xlabel("Outcome measures per trial"); ax[1,0].set_ylabel("Trials")
ax[1,0].set_title("(c) Outcomes per trial", loc="left")
ax[1,0].annotate(f"median {per_trial.median():.0f}, max {per_trial.max()}",
                 xy=(0.97, 0.9), xycoords="axes fraction", ha="right", fontsize=6.5)

# (d) text length
ax[1,1].hist(words, bins=40, color="#E69F00", edgecolor="black", linewidth=0.4)
ax[1,1].axvline(128, ls="--", lw=1, color="black")
ax[1,1].annotate("128-token budget", (128, ax[1,1].get_ylim()[1]*0.55),
                 fontsize=6.5, rotation=90, ha="right")
ax[1,1].set_xlabel("Words per outcome measure"); ax[1,1].set_ylabel("Outcome measures")
ax[1,1].set_title("(d) Text length", loc="left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(C.FIGS / f"fig01_corpus_overview.{ext}")
plt.close(fig)
print("figure ->", C.FIGS / "fig01_corpus_overview.pdf")
print(f"panel (b) spans {int(m['Biomarker'].min())} to {int(m['Biomarker'].max())} "
      "records, hence the log axis")

#%% md
## Vocabulary signal

Tokens most associated with each class, by log-odds ratio. Not used by any
model — this is descriptive, and it is what makes the later error analysis
interpretable: when a model fails, this table usually explains why.
#%% code
from sklearn.feature_extraction.text import CountVectorizer
cvz = CountVectorizer(min_df=5, ngram_range=(1, 2), stop_words="english")
M = cvz.fit_transform(df["clean_freeform"].fillna(""))
vocab = np.array(cvz.get_feature_names_out())
yv = df["has_biomarker"].values
cpos = np.asarray(M[yv == 1].sum(axis=0)).ravel() + 1
cneg = np.asarray(M[yv == 0].sum(axis=0)).ravel() + 1
lor = np.log((cpos / cpos.sum()) / (cneg / cneg.sum()))
top = pd.DataFrame({"token": vocab, "log_odds": lor,
                    "n_pos": cpos - 1, "n_neg": cneg - 1})
print("Most biomarker-associated tokens:")
print(top.nlargest(15, "log_odds").to_string(index=False))
print("\nMost non-biomarker-associated tokens:")
print(top.nsmallest(15, "log_odds").to_string(index=False))
C.save_table(top.sort_values("log_odds", ascending=False), C.TABLES / "vocabulary_log_odds.csv")
#%% md
## Done

**Next:** `02_classical_tiers.ipynb`.
