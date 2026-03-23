# figure_data_and_scripts

Data and plotting scripts for selected figure panels. Each subdirectory is self-contained.
All plotting scripts require Python 3 with `numpy`, `matplotlib`, `ase`, and `scipy`.
The shared `matplotlib.rc` style file must remain at this level (`../matplotlib.rc` relative to each panel directory).

---

## fig3_disregistry_1p1deg — Fig. 3d, 3e

**Generates:** Disregistry magnitude plots for MoS₂/WSe₂ at 1.1° twist angle, comparing reference, UM1, and UM2 relaxed structures.

**Data:**
- `MoS2-WSe2_1p1deg_reference.xyz` — DFT reference structure
- `MoS2-WSe2_1p1deg_UM1_relaxed.xyz`, `MoS2-WSe2_1p1deg_UM2_relaxed.xyz` — MLIP-relaxed structures
- `MoS2-WSe2_config_space_energies.xyz` — configuration-space energy sampling

**Run:**
```bash
python plot_fig3d.py          # fig3d_disregistry_magnitude.pdf
python plot_fig3e_UM1.py      # fig3e_disregistry_UM1.png
python plot_fig3e_UM2.py      # fig3e_disregistry_UM2.png
```

`compute_mean_disregistry_error.py` computes mean disregistry error statistics (used by `plot_fig3d.py`).

---

## fig3_rmse_vs_mde — Fig. 3f

**Generates:** Scatter plots of energy/force RMSE vs. mean disregistry error (MDE) for corrupted MLIPs (intralayer and interlayer).

**Data:**
- `interlayer_reference/energy_baseline_all_test_2_inter_reference.xyz` — DFT reference (interlayer test set)
- `intralayer_reference/energy_baseline_all_test_2_intra_reference.xyz` — DFT reference (intralayer test set)
- `MoS2-WSe2_interlayer_disregistry_distances.npy`, `MoS2-WSe2_intralayer_disregistry_distances.npy` — pre-computed Wasserstein MDE values

**Note:** The corrupted MLIP trajectory files (`interlayer_corrupted/`, `intralayer_corrupted/`) are not included due to size (~1 GB total). The script uses a load-or-compute cache: if the corrupted xyz files are present it reads them and caches the result as `.npy`; without them it cannot regenerate the RMSE axis. Corrupted trajectories can be regenerated using `corrupt_models.py` in `../surrogate_models/`.

**Run:**
```bash
./generate_figures.sh         # calls plot_fig3f.py
# outputs: fig3f_energy_rmse_vs_mde.pdf, fig3f_force_rmse_vs_mde.pdf
```

---

## fig4_q1D_vs_2D_mde — Fig. 4d

**Generates:** Bar/scatter comparison of MDE for q1D vs. 2D relaxed MoS₂/WSe₂ structures (mean and median variants).

**Data:**
- `MoS2-WSe2_1D_interlayer_disregistry_distances.npy`
- `MoS2-WSe2_1D_intralayer_disregistry_distances.npy`
- `MoS2-WSe2_2D_interlayer_disregistry_distances.npy`
- `MoS2-WSe2_2D_intralayer_disregistry_distances.npy`

**Run:**
```bash
./generate_figures.sh         # calls plot_fig4d.py
# outputs: fig4d_mde_mean.pdf, fig4d_mde_median.pdf
```

---

## fig6_2D_disregistry — Fig. 6d

**Generates:** 2D disregistry magnitude map for MLIP-relaxed GaS/HfS₂ moiré.

**Data:**
- `GaS-HfS2_2D_moire_MLIP_relaxed.xyz` — MLIP-relaxed 2D moiré structure

**Run:**
```bash
python plot_fig6d.py          # fig6d_2D_disregistry_magnitude_reference.png
```

---

## fig6_band_structures — Fig. 6b, 6c

**Data:** Quantum ESPRESSO input files and corresponding structures for GaS/HfS₂ band structure calculations at three geometries:

| File prefix | Geometry |
|---|---|
| `GaS-HfS2_initial_*` | Unrelaxed (initial) structure |
| `GaS-HfS2_MLIP_relaxed_*` | MLIP-relaxed structure |
| `GaS-HfS2_DFT_relaxed_*` | DFT-relaxed structure |

Each geometry has a `_structure.xyz` (atomic positions) and a `_bands.pwi` (QE bands calculation input). Run with `pw.x < <prefix>_bands.pwi`. Post-processing and plotting of the band structure outputs is not included here.
