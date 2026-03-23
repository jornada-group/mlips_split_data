#!/usr/bin/env bash
# Regenerate RMSE vs disregistry panels from bundled trajectories and cached .npy.
# Dependencies: Python 3 with numpy, matplotlib, and ase.
# Example: source /path/to/venv/bin/activate && ./generate_figures.sh
set -euo pipefail
cd "$(dirname "$0")"
exec python plot_fig3f.py
