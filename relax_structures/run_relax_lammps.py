from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures
import numpy as np  # Import numpy for numerical operations
from n_layer_calculator import  NLayerCalculator  # Import custom calculators from moirecompare package
from lammps_calculator import (MonolayerLammpsCalculator, 
                                      InterlayerLammpsCalculator) # Import custom calculators from moirecompare package
from ase.optimize import FIRE # Import optimization algorithms from ASE

def run_lammps_relax(input_file, output_file, layer_symbols, intralayer_potential='tmd.sw', interlayer_potential='WS.KC'):
    from ase.io import read, write  # Import read and write functions from ASE for handling atomic structures

    # Read the input atomic structure
    atoms = read(input_file, format="extxyz")
    
    # Initialize a list of intralayer calculators
    intralayer_calcs = [
        MonolayerLammpsCalculator(atoms[atoms.arrays['atom_types'] < 3],
                                  layer_symbols[0],
                                  system_type='TMD',
                                  intra_potential=intralayer_potential)
    ]
    # Initialize a list of interlayer calculators
    interlayer_calcs = []

    # Loop through the layers and set up calculators
    for i in np.arange(1, len(layer_symbols)):
        layer_atoms = atoms[
            np.logical_and(atoms.arrays['atom_types'] >= i * 3,
                           atoms.arrays['atom_types'] < (i + 1) * 3)
        ]
        print(np.unique(layer_atoms.arrays['atom_types']))  # Print unique atom types in the current layer
        intralayer_calcs.append(MonolayerLammpsCalculator(layer_atoms,
                                                          layer_symbols=layer_symbols[i],
                                                          system_type='TMD',
                                                          intra_potential=intralayer_potential))

        bilayer_atoms = atoms[np.logical_and(atoms.arrays['atom_types'] >= (i - 1) * 3,
                                             atoms.arrays['atom_types'] < (i + 1) * 3)]
        print(np.unique(bilayer_atoms.arrays['atom_types']))  # Print unique atom types in the bilayer
        print(layer_symbols[i - 1:i + 1])  # Print symbols for the current bilayer

        interlayer_calcs.append(
            InterlayerLammpsCalculator(bilayer_atoms,
                                       layer_symbols=layer_symbols[i - 1:i + 1],
                                       system_type='TMD',
                                       inter_potential=interlayer_potential))

    # Combine the intra- and interlayer calculators into an NLayerCalculator
    n_layer_calc = NLayerCalculator(atoms,
                                    intralayer_calcs,
                                    interlayer_calcs,
                                    layer_symbols)

    # Assign the combined calculator to the atoms object
    atoms.calc = n_layer_calc

    # Perform an initial calculation to get unrelaxed energy
    atoms.calc.calculate(atoms)
    print(f"Unrelaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    # Set up the FIRE optimizer for structural relaxation
    dyn = FIRE(atoms, trajectory=f'{output_file}.traj')
    dyn.run(fmax=1e-3)  # Run until the maximum force is below 1e-3 eV/Å

    # Print the relaxed energy
    print(f"Relaxed: Total_energy {atoms.calc.results['energy']:.3f} eV, \n",
          f"layer_energy {atoms.calc.results['layer_energy']}")

    from ase.io.trajectory import Trajectory  # Import Trajectory module from ASE for handling trajectories

    # Read the trajectory file generated during relaxation
    traj_path = f"{output_file}.traj"
    traj = Trajectory(traj_path)
    images = [atom for atom in traj]  # Collect all images from the trajectory

    # Write the final relaxed structure to an output file in extxyz format
    write(f"{output_file}.traj.xyz", images, format="extxyz")

# Example usage of the function
if __name__ == "__main__":
    # Define the input XYZ file and output file prefix
    xyz_file_path = "test_structures/MoS2_WSe2_2D.xyz"
    out_file = 'test_structures/MoS2_WSe2_2D_lammps'
    
    # Define the intralayer and interlayer potentials
    intralayer_potential = 'potentials/tmd.sw'
    interlayer_potential = 'potentials/WS.KC'

    # Define the layer symbols
    layer_symbols = [["Mo", "S", "S"],
                     ["W", "Se", "Se"]]

    # Call the relaxation function with specified input and output paths
    run_lammps_relax(xyz_file_path, out_file, layer_symbols, intralayer_potential, interlayer_potential)