# Structure Relaxation Toolkit

This directory contains scripts for relaxing atomic structures of layered materials using both MLIPs (Allegro) and classical force fields (SW+KC with LAMMPS).

## Directory Structure

- `allegro_calculator.py`: Interlayer and intralayer ASE calculator implementing the Allegro machine learning potential
- `lammps_calculator.py`: Interlayer and intralayer ASE calculators implementing LAMMPS-based classical force fields
- `n_layer_calculator.py`: Meta-calculator for handling multi-layer structures combining intralayer and interlayer calculators
- `run_relax_allegro.py`: Script for running relaxations using the Allegro calculator
- `run_relax_lammps.py`: Script for running relaxations using the LAMMPS calculator
- `test_structures/`: Directory containing example structures for testing
- `potentials/`: Directory containing potential files for both Allegro and LAMMPS

## Calculator Files

### allegro_calculator.py

Implements the `AllegroCalculator` class, which is an ASE-compatible calculator for interlayer and intralayer interactions using Allegro MLIPs.

### lammps_calculator.py

Contains two primary calculator classes:

1. `MonolayerLammpsCalculator`: ASE calculator for intralayer interactions in monolayer systems
   - Uses Stillinger-Weber potentials for intralayer interactions
   
2. `InterlayerLammpsCalculator`: ASE calculator for interlayer interactions in layered systems
   - Handles interactions between stacked 2D materials
   - Uses Kolmogorov-Crespi potentials for interlayer interactions

### n_layer_calculator.py

Implements the `NLayerCalculator` class, which:

- Manages both intralayer and interlayer interactions
- Separates atoms into layers based on atom_types
- Allows different calculators for different types of interactions
- Combines results from all calculators to compute total energies and forces

## Potentials and Test Structures

### Potentials

Located in the `potentials/` directory:

- **Allegro Machine Learning Models**:
  - `mos2_intra.pth`: Intralayer Allegro model for MoS2
  - `wse2_intra.pth`: Intralayer Allegro model for WSe2
  - `WSeMoS_inter.pth`: Interlayer Allegro model for WSe2-MoS2 interfaces

- **Classical Force Fields**:
  - `tmd.sw`: Stillinger-Weber potential for TMD materials (intralayer)
  - `WS.KC`: Kolmogorov-Crespi potential for TMD materials (interlayer)

### Test Structures

Located in the `test_structures/` directory:

- `MoS2_WSe2_2D.xyz`: Example heterostructure of MoS2 and WSe2 in 2D: 2973 atoms

## Running Relaxations

### Using Allegro (Machine Learning)

The `run_relax_allegro.py` script demonstrates how to use the Allegro calculator for structure relaxation:

The script will:
1. Load the input structure from the specified XYZ file
2. Set up the appropriate calculators using the specified models
3. Perform relaxation using the FIRE algorithm
4. Save the relaxed structure and trajectory to output files

### Using LAMMPS (Classical Force Fields)

The `run_relax_lammps.py` script demonstrates how to use the LAMMPS calculators for structure relaxation:

The script will:
1. Load the input structure from the specified XYZ file
2. Set up the appropriate LAMMPS calculators using the specified potentials
3. Perform relaxation using the FIRE algorithm
4. Save the relaxed structure and trajectory to output files

## Requirements

- ASE (Atomic Simulation Environment)
- NEquIP/Allegro (for the Allegro calculator)
- LAMMPS (for the LAMMPS calculator)
- NumPy
- PyTorch (for the Allegro calculator) 