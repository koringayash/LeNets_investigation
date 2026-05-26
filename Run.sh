#!/usr/bin/env bash
# =============================================================================
# Run.sh — CV Framework v2
#
# Usage
# -----
#   bash Run.sh                              # full pipeline, all 3 stages
#   bash Run.sh --resume                     # resume incomplete stages
#   bash Run.sh --stage training             # single stage, fresh
#   bash Run.sh --stage training --resume    # resume training only
#   bash Run.sh --epochs 5                   # override epoch count
#
# VM tip: nohup bash Run.sh > run.log 2>&1 &
#         tail -f logs/main.log
# =============================================================================

set -e

GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
print_step() { echo -e "\n${BLUE}══════════════════════════════════════${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}══════════════════════════════════════${NC}\n"; }
print_ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_step "Step 1/2 — Installing dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
print_ok "Dependencies installed"

print_step "Step 2/2 — Running pipeline"
echo "  Arguments: $@"
python main.py "$@"

echo ""
print_ok "Pipeline complete!"
echo -e "  ${GREEN}Task        →${NC} $(python -c 'from config import EXPERIMENT; print(EXPERIMENT[\"task\"])')"
echo -e "  ${GREEN}Logs        →${NC} logs/"
echo -e "  ${GREEN}Checkpoints →${NC} Checkpoint/"
echo -e "  ${GREEN}Plots       →${NC} plots/"
echo ""
echo "  Follow live logs : tail -f logs/main.log"
echo "  Resume training  : bash Run.sh --stage training --resume"