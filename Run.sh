#!/usr/bin/env bash
# =============================================================================
# Run.sh — CV Framework master run script
#
# Usage
# -----
#   bash Run.sh                          # full pipeline from scratch
#   bash Run.sh --resume                 # resume all incomplete stages
#   bash Run.sh --stage dataset          # single stage, fresh
#   bash Run.sh --stage training --resume  # resume training only
#   bash Run.sh --epochs 5               # override epoch count
#
# All arguments are forwarded to main.py, so any flag main.py accepts
# works here too.
#
# VM tip: use `nohup bash Run.sh > run.log 2>&1 &` to run in background
# and safely disconnect. Then `tail -f run.log` to check progress.
# =============================================================================

set -e  # exit immediately on any error

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'

print_step() { echo -e "\n${BLUE}══════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}══════════════════════════════════════${NC}\n"; }
print_ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_step "Step 1/3 — Installing dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
print_ok "Dependencies installed"

print_step "Step 2/3 — Sanity check (dataset stage dry-run)"
python main.py --stage dataset
print_ok "Dataset stage passed"

print_step "Step 3/3 — Running pipeline"
echo "  Arguments: $@"
python main.py "$@"

echo ""
print_ok "Pipeline complete!"
echo -e "  ${GREEN}Logs        →${NC} logs/"
echo -e "  ${GREEN}Checkpoints →${NC} Checkpoint/"
echo -e "  ${GREEN}Plots       →${NC} plots/"
echo ""
echo "  Follow live logs: tail -f logs/main.log"