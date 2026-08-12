#!/usr/bin/env bash
# run_demo.sh -- Launch the interactive Gradio demo.
#
# The demo requires models/metadata.json to exist. This file is written
# automatically by step 4 (04_train_zeroshot_llm.py).
# If it is missing, run the pipeline first.
#
# Usage:
#   bash run_demo.sh            # local only (http://localhost:7860)
#   bash run_demo.sh --share    # generate a public Gradio link

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "models/metadata.json" ]; then
    echo ""
    echo "ERROR: models/metadata.json not found."
    echo ""
    echo "The demo requires trained models to be saved first."
    echo "Run the full pipeline (steps 01-05) with:"
    echo "  bash run_pipeline.sh"
    echo ""
    exit 1
fi

echo ""
echo "Starting Biomarker Detection Demo..."
echo ""

if [ "$1" == "--share" ]; then
    echo "Public sharing enabled. A public URL will be printed below."
    echo ""
    conda run -n biomarker-ad python demo/app.py --share
else
    echo "Running locally at http://localhost:7860"
    echo "Use 'bash run_demo.sh --share' to generate a public link."
    echo ""
    conda run -n biomarker-ad python demo/app.py
fi
