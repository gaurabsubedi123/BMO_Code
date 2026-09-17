"""
Recall by biomarker modality.

BMO_PARTITION=oof (default) uses out-of-fold predictions over the training
partition, where each record was predicted by a model that never saw it. The
held-out partition contains too few positive records per modality to estimate
recall -- digital assessments have exactly one -- so the out-of-fold estimate is
the one the manuscript quotes. BMO_PARTITION=test uses the held-out partition.

The manuscript reports that recall is high on conventional modalities but falls
sharply on digital assessments, and that digital assessments supply a small share
of the biomarker class but a large share of false negatives. Those figures had no
source file; this derives them.

The modality mapping is copied verbatim from 01_dataset_and_corpus so the
taxonomy is identical to the one behind table_modality_composition.csv. Recall is
computed per encoder and then averaged across the nine fine-tuned encoders, which
is the quantity the manuscript quotes.

Writes $BMO_RUN_DIR/tables/recall_by_modality.csv
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

RUN = Path(os.environ.get("BMO_RUN_DIR", "/mnt/subedi-research/projects/bmo_rerun_2026"))

MRS_UD = {"UD_GSH", "UD_NAD", "UD_NAD/NADH", "UD_CK/ATPase"}   # in-vivo spectroscopy
MRI_UD = {"UD_AFD", "MT_Myelin", "CDM"}


def modality(annotation):
    a = str(annotation).strip()
    first = a.upper().split(",")[0].strip()
    if first.startswith("ANALOG"):                    return "Clinical / behavioural"
    if a in MRS_UD or a in MRI_UD:                    return "Imaging (MRI/MRS)"
    if first.startswith(("RETINAL", "OCT")):          return "Imaging (retinal OCT)"
    if first.startswith("TMS"):                       return "Neurophysiology"
    if first.startswith("EXOSOME"):                   return "Fluid (exosomal)"
    if first.startswith("STOOL"):                     return "Microbiome / stool"
    if re.match(r"^(Aβ|AB|TAU|FDG|SV2A|TSPO)", first) or "PET" in first:
        return "Imaging (PET)"
    if any(k in first for k in ("MRI", "ROI", "NODDI", "DTI", "MRS", "ASL", "FMRI", "WMH", "VOLUM")):
        return "Imaging (MRI/MRS)"
    if first.startswith(("CSF", "PLASMA", "SERUM", "SEURM", "FLUID", "BLOOD", "URINE", "SALIVA")):
        return "Fluid"
    if first.startswith("UD_"):                       return "Fluid"
    if any(k in first for k in ("EEG", "PSG", "MEG", "ERP", "EMG", "ECG", "EYE")):
        return "Neurophysiology"
    if first.startswith("DIGITAL") or "ACTIGRAPH" in first:  return "Digital"
    if any(k in first for k in ("APOE", "GENOT", "GENE", "SNP", "POLYGEN", "RNA", "DNA", "OMIC")):
        return "Genetic / omics"
    if any(k in first for k in ("PK", "ADA", "ANTIBOD", "IMMUNOGEN")):
        return "Pharmacokinetic / immunogenicity"
    if first == "COGNITION":                          return "Clinical / behavioural"
    return "Unclassified"


ds = pd.read_csv(RUN / "outputs/dataset/labeled_dataset.csv")
PART = os.environ.get("BMO_PARTITION", "oof")   # "oof" (train, out-of-fold) or "test"
idx = np.load(RUN / ("outputs/dataset/%s_idx.npy" % ("train" if PART=="oof" else "test")))
te = ds.loc[idx].reset_index(drop=True)
te["modality"] = te["Annotation"].fillna("(none)").map(modality)

pf = "oof_predictions_merged.csv" if PART=="oof" else "test_predictions_merged.csv"
preds = pd.read_csv(RUN / "outputs/bert_models" / pf)
assert len(preds) == len(te), f"row mismatch: {len(preds)} predictions vs {len(te)} records"
y = te["has_biomarker"].to_numpy().astype(int)
assert (preds["y_true"].to_numpy().astype(int) == y).all(), \
    "prediction rows are not aligned with the test partition"

encoders = [c for c in preds.columns if c != "y_true"]
print(f"{PART} partition {len(te)} records | {len(encoders)} encoders: {', '.join(encoders)}")

pos = te["has_biomarker"] == 1
rows = []
for mod, grp in te[pos].groupby("modality"):
    sel = grp.index.to_numpy()
    recalls = [float((preds.loc[sel, e].to_numpy().astype(int) == 1).mean()) for e in encoders]
    fn_counts = [int((preds.loc[sel, e].to_numpy().astype(int) == 0).sum()) for e in encoders]
    total_fn = [int(((preds[e].to_numpy().astype(int) == 0) & (y == 1)).sum()) for e in encoders]
    rows.append(dict(
        Modality=mod,
        biomarker_records=int(len(sel)),
        pct_of_biomarker_class=round(100 * len(sel) / int(pos.sum()), 2),
        mean_recall=round(float(np.mean(recalls)), 4),
        min_recall=round(float(np.min(recalls)), 4),
        max_recall=round(float(np.max(recalls)), 4),
        mean_false_negatives=round(float(np.mean(fn_counts)), 2),
        pct_of_all_false_negatives=round(
            100 * float(np.mean(fn_counts)) / float(np.mean(total_fn)), 2),
        n_encoders=len(encoders)))

out = (pd.DataFrame(rows).sort_values("mean_recall", ascending=False)
       .reset_index(drop=True))
dest = RUN / ("tables/recall_by_modality_%s.csv" % PART)
out.to_csv(dest, index=False)
print()
print(out.to_string(index=False))
print(f"\nwrote {dest}")
