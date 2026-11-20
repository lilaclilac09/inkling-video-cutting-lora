#!/usr/bin/env bash
# Walkthrough 1: install the harness and verify it, CPU only, no Tinker spend.
source "$(dirname "$0")/lib.sh"
say "1. Install the harness (no GPU, no Tinker account needed)"
run "uv sync"
say "2. Run the tests: contract, scoring, eval integrity, cutter adapter"
run "uv run pytest"
say "3. Inspect the 31 approved picks the LoRA learned from"
run "uv run python prepare.py --audit"
say "4. The scorer must give the references a perfect 1.0"
run "uv run python prepare.py --self-check"
say "5. Dry-run the whole training loop with a stub model (writes nothing, spends nothing)"
run "uv run python train.py --dry-run"
say "Harness verified. Next: turn model picks into clips (walkthrough 2)."
