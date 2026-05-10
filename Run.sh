#!/usr/bin/env bash
# =============================================================================
# Run.sh
# ------
# Master script for the LeNet Empirical Study.
#
# What this script does
# ---------------------
#   1. Installs Python dependencies from requirements.txt.
#   2. Runs a quick dry-run (1 batch) to verify the pipeline is working.
#   3. Runs all 8 experiments sequentially (8 activation × pooling combos).
#   4. Prints the location of the results CSV at the end.
#
# Usage
# -----
#   bash Run.sh                          # full study (all 8 experiments)
#   bash Run.sh --dry-run                # sanity check only (1 batch each)
#   bash Run.sh --experiment lenet_relu_maxpool   # single experiment
#   bash Run.sh --epochs 5               # override epoch count
#
# All arguments after 'Run.sh' are forwarded to train.py, so any argument
# that train.py accepts works here too.
#
# Notes for VM users
# ------------------
# - All training output is mirrored to log files under /logs/ so you can
#   disconnect from the VM and check progress later.
# - Use `tail -f logs/main.log` to follow the master log live.
# - Use `tail -f logs/lenet_relu_maxpool.log` to follow a specific experiment.
# =============================================================================

set -e  # exit immediately if any command fails

# ---- Colour helpers (degrade gracefully if terminal doesn't support them) --
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'   # No Colour

print_step() {
    echo -e "\n${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}\n"
}

print_ok() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_warn() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

# ---- Resolve project root (the directory this script lives in) -------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 1. Install dependencies ------------------------------------------------
print_step "Step 1/3 — Installing dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
print_ok "Dependencies installed."

# ---- 2. Dry-run (sanity check) ---------------------------------------------
print_step "Step 2/3 — Dry-run (pipeline sanity check)"
python3 train.py --dry-run --epochs 1
print_ok "Dry-run passed. Pipeline is healthy."

# ---- 3. Full training -------------------------------------------------------
print_step "Step 3/3 — Running the full empirical study"
echo "  Arguments forwarded to train.py: $@"
echo ""

python3 train.py "$@"

# ---- Done ------------------------------------------------------------------
echo ""
print_ok "All experiments finished!"
echo ""
echo -e "  ${GREEN}Results CSV   →${NC} logs/results.csv"
echo -e "  ${GREEN}Checkpoints   →${NC} Checkpoint/"
echo -e "  ${GREEN}Log files     →${NC} logs/"
echo ""
echo "  To inspect results:"
echo "    cat logs/results.csv"
echo "    column -t -s, logs/results.csv   # pretty-print the CSV"
echo ""
