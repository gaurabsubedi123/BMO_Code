"""Does batching with left padding change Qwen3.8's scores? Re-score test records one
at a time and compare with the saved batch-of-8 probabilities from the full run."""
import sys, numpy as np, pandas as pd, torch
sys.argv = [sys.argv[0]]
import qwen38_zeroshot as Q
REV = open("/mnt/subedi-research/projects/bmo_qwen38_2026/logs/revision.txt").read().strip()
saved = pd.read_csv(Q.OUT / "test_predictions.csv")
_, _, texts_te, labels_te = Q.load_splits()
assert (saved.y_true.values == labels_te).all()
tok, model = Q.load_model(Q.CHECKPOINTS["bf16"], "bf16", REV)
p1 = Q.score(model, tok, texts_te, "direct", batch_size=1)
d = np.abs(p1 - saved.p_yes.values)
flip05 = int(((p1 >= 0.5) != (saved.p_yes.values >= 0.5)).sum())
flip10 = int(((p1 >= 0.1) != (saved.p_yes.values >= 0.1)).sum())
print(f"all 341 test records, batch 1 vs saved batch 8: max |dp| {d.max():.4g}, mean {d.mean():.3g}, "
      f"records with |dp|>0.01: {(d > 0.01).sum()}")
print(f"decision flips at 0.5: {flip05}, at 0.1 (tuned): {flip10}")
e1 = int(((p1 >= 0.5).astype(int) != labels_te).sum()); e2 = int(((p1 >= 0.1).astype(int) != labels_te).sum())
print(f"errors with batch-1 scores: default {e1}, tuned {e2}")
pd.DataFrame({"p_yes_batch8": saved.p_yes.values, "p_yes_batch1": p1}).to_csv(
    Q.OUT / "padding_check_test.csv", index=False)
print("PADCHECK DONE")
