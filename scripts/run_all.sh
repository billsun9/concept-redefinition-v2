#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-config/smoke.yaml}
python scripts/make_dataset.py "$CFG"
python scripts/run_behavior.py "$CFG"
python scripts/collect_activations.py "$CFG"
python scripts/analyze_movement.py "$CFG"
python scripts/train_probe.py "$CFG"
python scripts/patch_activations.py "$CFG"
python scripts/vocab_effects.py "$CFG"
echo "All done for $CFG"
