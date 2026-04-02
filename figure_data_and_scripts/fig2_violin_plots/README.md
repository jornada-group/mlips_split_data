# Figure 2 — Violin Plots of Model Accuracy

This folder contains the data and plotting script to reproduce Figure 2 of the paper, which compares the accuracy (energy and force residuals) of various interatomic potential models for MoS2/WSe2 bilayer systems.

## Contents

```
fig2_violin_plots/
├── plot_fig2.py           # Plotting script for Figure 2
├── plot_fig_si2.py        # Plotting script for Supplementary Figure SI-2
├── matplotlib.rc          # Matplotlib style configuration
├── fig_2.pdf           # Figure 2 output (PDF)
├── fig_2.png           # Figure 2 output (PNG)
├── fig_si_2.pdf           # SI Figure 2 output (PDF)
├── fig_si_2.png           # SI Figure 2 output (PNG)
└── data/
    ├── baseline_all_test_2.xyz        # DFT reference test set
    ├── intra_WSe2/                    # Intralayer model predictions
    │   ├── baseline_01_all_valid.xyz  # Unified Allegro (rcut = 6.0 Å)
    │   ├── baseline_02_all_valid.xyz  # Unified Allegro (rcut = 10.0 Å)
    │   ├── split_intra_03_small_all_valid.xyz  # Split Allegro intralayer
    │   ├── split_mace_intra_wse2.xyz  # Split MACE intralayer
    │   └── sw_lammps_wse2_validation.xyz       # Stillinger-Weber baseline
    └── inter_MoS2WSe2/                # Interlayer model predictions
        ├── baseline_01_all_valid.xyz  # Unified model
        ├── kc_lammps_mos2_wse2_validation.xyz  # Kolmogorov-Crespi baseline
        ├── split_inter_01_all_valid.xyz
        ├── split_inter_02_small_all_valid.xyz  # Split Allegro interlayer (small NN)
        └── split_inter_03_all_valid.xyz        # Split Allegro interlayer (large NN)
```

## What the figures show

**Figure 2** (`plot_fig2.py`): Two-panel (2×1) violin plot comparing intralayer and interlayer energy-per-atom residuals. Models shown: random guessing, Stillinger-Weber / Kolmogorov-Crespi classical potentials, and Allegro unified/split models at two cutoff radii or network sizes.

**SI Figure 1** (`plot_fig_si2.py`): Four-panel (2×2) violin plot extending Figure 2 to include both energy and force residuals, and adds the Split MACE model as an additional comparison point for both intra- and interlayer components.

Each violin represents the full distribution of absolute residual errors over the DFT test set. Horizontal lines indicate target accuracy thresholds.

## Reproducing the figures

```bash
python plot_fig2.py       # generates fig_2_v8.pdf and fig_2_v8.png
python plot_fig_si2.py    # generates fig_si_2.pdf and fig_si_2.png
```

**Dependencies**: `numpy`, `ase`, `matplotlib`

**Optional**: If you have the `Helvetica Neue LT Com` font installed in `~/fonts/`, it will be loaded automatically. Otherwise matplotlib will fall back to its default sans-serif font.
