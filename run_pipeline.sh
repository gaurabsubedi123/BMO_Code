#!/usr/bin/env bash
# run_pipeline.sh -- Run the complete BMO classification pipeline end to end.
#
# This script executes all 7 steps in order. If any step fails, the script
# stops immediately (set -e) so you can diagnose the problem before wasting
# time on downstream steps.
#
# Estimated total runtime (with GPU):
#   Step 1:   ~2 minutes   (prepare dataset, create 80/20 split)
#   Step 2:   ~10 minutes  (classical ML nested CV + test eval + confusion matrices)
#   Step 2a:  ~30-60 min   (optional: TF-IDF+FastText+MLP+SHAP; requires gensim, shap)
#   Step 2b:  ~30-60 min   (optional: TF-IDF+BioWordVec+MLP+SHAP; requires fasttext, shap)
#   Step 3:   ~6-8 hours   (BERT CV + demo weights + test eval + OOF CMs)
#   Step 4:   ~30-60 min   (zero-shot inference + CMs + test eval)
#   Step 5:   ~2 minutes   (consolidate all results, comparison figures)
#   Step 6:   ~5-10 min    (unified held-out test set evaluation, 18-36 models)
#   Step 7:   ~1-2 min     (McNemar's pairwise significance tests, 153-630 pairs)
#
# Usage:
#   bash run_pipeline.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tee all stdout and stderr to train_report.txt while still showing in terminal.
REPORT_FILE="train_report.txt"
exec > >(tee -a "$REPORT_FILE") 2>&1

echo ""
echo "--------------------------------------------------------------"
echo "BMO Classification Pipeline"
echo "Run started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Log file   : $SCRIPT_DIR/$REPORT_FILE"
echo "--------------------------------------------------------------"
echo ""

# echo "Step 1 of 7: Preparing dataset..."
# conda run -n biomarker-ad python src/01_prepare_dataset.py
# echo ""

# echo "Step 2 of 7: Training classical ML (CV + test eval + confusion matrices)..."
# conda run -n biomarker-ad python src/02_train_classical_ml.py
# echo ""

echo "Step 2a of 7 (optional): Enhanced Classical ML (TF-IDF + FastText + MLP + SHAP)..."
if conda run -n biomarker-ad python -c "import gensim, shap" 2>/dev/null; then
    conda run -n biomarker-ad python src/02a_classical_ml_enhanced.py
    echo ""
else
    echo "  Skipping — gensim and/or shap not installed in biomarker-ad environment."
    echo "  To enable: conda run -n biomarker-ad pip install gensim shap"
    echo ""
fi

echo "Step 2b of 7 (optional): BioWordVec Classical ML (TF-IDF + BioWordVec + MLP + SHAP)..."
if conda run -n biomarker-ad python -c "import fasttext, shap" 2>/dev/null; then
    conda run -n biomarker-ad python src/02b_classical_ml_biowordvec.py
    echo ""
else
    echo "  Skipping — fasttext and/or shap not installed in biomarker-ad environment."
    echo "  To enable: conda run -n biomarker-ad pip install fasttext-wheel shap"
    echo "  Also download: BioWordVec_PubMed_MIMICIII_d200.bin (~26 GB)"
    echo ""
fi

echo "Step 3 of 7: Training BERT models (CV + demo weights + test eval + OOF CMs)..."
conda run -n biomarker-ad python src/03_train_bert_models.py
echo ""

echo "Step 4 of 7: Zero-shot LLM evaluation (inference + CMs + test eval)..."
conda run -n biomarker-ad python src/04_train_zeroshot_llm.py
echo ""

echo "Step 5 of 7: Consolidating all results into comparison figures..."
conda run -n biomarker-ad python src/05_consolidate_results.py
echo ""

echo "Step 6 of 7: Unified held-out test set evaluation (18+ models, enhanced/biowordvec if available)..."
conda run -n biomarker-ad python src/06_test_set_evaluation.py
echo ""

echo "Step 7 of 7: McNemar's pairwise significance tests (153+ pairs, up to 630 if 02a+02b run)..."
conda run -n biomarker-ad python src/07_significance_tests.py
echo ""

echo "--------------------------------------------------------------"
echo "Pipeline complete."
echo "Run finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Full log saved to: $SCRIPT_DIR/$REPORT_FILE"
echo ""
echo "Key outputs:"
echo "  outputs/dataset/labeled_dataset.csv              - the training dataset"
echo "  outputs/classical_ml/fig_cm_oof.png              - classical ML OOF confusion matrices"
echo "  outputs/classical_ml/fig_fold_stability.png      - fold-by-fold F1 (stability check)"
echo "  outputs/classical_ml/fig_test_eval.png           - classical ML test set evaluation"
echo "  outputs/bert_models/fig_loss_curves.png          - BERT training/validation loss curves"
echo "  outputs/bert_models/fig_cm_oof.png               - BERT OOF confusion matrices"
echo "  outputs/bert_models/fig_test_eval.png            - BERT test set evaluation"
echo "  outputs/zeroshot_llm/fig_cm.png                  - zero-shot confusion matrices"
echo "  outputs/zeroshot_llm/fig_test_eval.png           - zero-shot test set evaluation"
echo "  outputs/final_results/master_results.csv             - all model CV scores"
echo "  outputs/final_results/results_table.tex              - LaTeX table for paper"
echo "  outputs/final_results/fig_all_models.png             - comparison figure"
echo "  outputs/final_results/test_set_results.csv           - held-out test set scores"
echo "  outputs/final_results/significance_tests.csv         - McNemar p-values (105+ pairs)"
echo "  outputs/final_results/fig_significance_heatmap.png   - significance heatmap"
echo "  outputs/final_results/significance_summary.txt       - significance narrative"
echo "  outputs/final_results/fig_test_set_confusion_matrices.png - all 15 models CMs"
echo "  models/                                              - saved model weights"
echo ""
echo "To run the interactive demo:"
echo "  bash run_demo.sh"
echo "--------------------------------------------------------------"
