# Supporting Data for "*Accurate, transferable, and verifiable machine-learned interatomic potentials for layered materials*"

This repository contains information and codes to reproduce the results in
- Georgaras, Ramdas, Shan, Halsted, Berwyn, Li, Jornada, *Accurate, transferable, and verifiable machine-learned interatomic potentials for layered materials*, [arXiv:2503.15432](https://arxiv.org/abs/2503.15432).

Customizations to the MACE machine-learned interatomic potential (MLIP) to handle interlayer interactions are provided [in an external repository](https://github.com/jornada-group/mace-interlayer/tree/akash_interlayer_modify?tab=readme-ov-file#interlayer-mace). No customization is necessary for the Allegro MLIP for interlayer interactions, though we mark the version of Allegro we used for our calculations [here](https://github.com/jornada-group/allegro).

## About the Repository
The `allegro_config_files` contains the config files that were used to train the Allegro models. The models for Figure 2 and Figure S1 were trained on the older Allegro version, and their corresponding YAML files are found in `allegro_config_files/models_fig2`. All the interlayer models were trained on the jornada_group fork of allegro found [here](https://github.com/jornada-group/allegro). The files have been modified to point to the corresponding dataset in al `allegro_config_files/datasets`. The `surrogate_models` folder contain the pth files used to produce the plots in Figures 3, 4. We share the pth here, as the weights are important in reproducing the results of the paper, to load them please ensure to be using pytorch 2.0 or later. The script used to corrupt the models can be found in `surrogate_models/corrupt_models.py`. The interlayer yaml file used to relax the HfS2/GaS bilayer is found in  `allegro_config_files/models_HfS2_GaS`

The `relax_structures` directory contains a calculators and example scripts for relaxing atomic structures of layered materials. It provides ASE-compatible calculators for both MLIPs (Allegro) and classical force fields (SW+KC with LAMMPS). The directory includes example scripts for structure relaxation, test structures, and necessary potential files. These tools were used to perform the relaxations described in the paper and can be used to reproduce the results or to relax new structures.

## Training your own interlayer models
We recommend that you currently use the MACE interlayer model training procedure available in the jornada_group [github](https://github.com/jornada-group/mace-interlayer/tree/akash_interlayer_modify?tab=readme-ov-file#interlayer-mace). We will release nequip and allegro versions compatible with their current develop branch shortly.

## Structure Relaxation
The `relax_structures` directory provides tools for relaxing layered material structures using both MLIPs (with Allegro)and classical force fields (with LAMMPS). See the README.md in that directory for detailed instructions on:
- Using the provided calculators for interlayer and intralayer interactions
- Running relaxations with Allegro (MLIP) or LAMMPS (SW+KC)
- Working with the example structures and potentials
- Implementing your own relaxation workflows

## Figure Data and Scripts
The `figure_data_and_scripts` directory contains the data and plotting scripts needed to reproduce the following figure panels from the paper:

- **fig3_disregistry_1p1deg** (Fig. 3d, 3e) — Disregistry magnitude plots for MoS₂/WSe₂ at 1.1° twist angle comparing reference, UM1, and UM2 relaxed structures. Run `plot_fig3d.py`, `plot_fig3e_UM1.py`, `plot_fig3e_UM2.py`.
- **fig3_rmse_vs_mde** (Fig. 3f) — Energy/force RMSE vs. mean disregistry error for corrupted MLIPs. Run `./generate_figures.sh`. Note: requires corrupted MLIP trajectory files (not included due to size; regenerate using `surrogate_models/corrupt_models.py`).
- **fig4_q1D_vs_2D_mde** (Fig. 4d) — MDE comparison for q1D vs. 2D relaxed MoS₂/WSe₂ structures. Run `./generate_figures.sh`.
- **fig6_2D_disregistry** (Fig. 6d) — 2D disregistry magnitude map for MLIP-relaxed GaS/HfS₂ moiré. Run `python plot_fig6d.py`.
- **fig6_band_structures** (Fig. 6b, 6c) — Quantum ESPRESSO input files and structures for GaS/HfS₂ band structure calculations at three geometries (unrelaxed, MLIP-relaxed, DFT-relaxed).

See `figure_data_and_scripts/README.md` for full details.
