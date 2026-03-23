#!/usr/bin/env bash
# Regenerate 1D vs 2D mean disregistry error comparison panels from bundled .npy inputs.
# Dependencies: Python 3 with numpy, matplotlib.
# Example: source /path/to/venv/bin/activate && ./generate_figures.sh
set -euo pipefail
cd "$(dirname "$0")"
exec python plot_fig4d.py
