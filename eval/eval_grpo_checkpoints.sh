#!/bin/bash
# Score the GRPO checkpoints (steps 20/40/60) on eval-set-v2 and re-score.
# Paths are read from eval/grpo-run.log (CHECKPOINT lines), so this runs
# hands-free right after the GRPO loop finishes.
# Usage: bash eval/eval_grpo_checkpoints.sh
# First: uv sync --extra train; export TINKER_API_KEY in your environment.
set -e
cd "$(dirname "$0")/.."
PY="uv run python"
for step in 20 40 60; do
  path=$(grep "CHECKPOINT step $step:" eval/grpo-run.log | sed 's/.*path=//')
  if [ -z "$path" ]; then
    echo "no checkpoint recorded for step $step — skipping"
    continue
  fi
  echo "=== grpo-$step: $path ==="
  $PY eval/run_eval.py --eval-path eval/eval-set-v2.jsonl \
    --model custom --model-path "$path" --out "eval/v2-grpo-$step.json"
done
$PY eval/rescore_v2.py eval/v2-grpo-20.json eval/v2-grpo-40.json eval/v2-grpo-60.json
