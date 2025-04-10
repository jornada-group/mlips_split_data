from ase.calculators.calculator import (
    Calculator,
    CalculatorError,
    CalculatorSetupError,
    all_changes,
)
from ase.data import atomic_masses, atomic_numbers, chemical_symbols
from nequip.scripts.deploy import load_deployed_model, R_MAX_KEY
from nequip.data import AtomicData, AtomicDataDict
from typing import List, Dict, Union, Tuple
from re import compile, Pattern, Match
from torch.jit import ScriptModule
from torch import long
import torch
from pathlib import Path
from ase.atoms import Atoms
from itertools import combinations
from numpy import zeros, where, array, logical_and
import numpy as np


torch.manual_seed(42)


def get_results_from_model_out(model_out):
    """
    Extracts results from model output dictionary.

    Parameters
    ----------
    model_out : dict
        Dictionary containing model outputs.

    Returns
    -------
    dict
        Dictionary containing energy, energies, and forces extracted from model_out.
    """
    results = {}
    if AtomicDataDict.TOTAL_ENERGY_KEY in model_out:
        results["energy"] = (
            model_out[AtomicDataDict.TOTAL_ENERGY_KEY]
            .detach()
            .cpu()
            .numpy()
            .reshape(tuple())
        )
        results["free_energy"] = results["energy"]
    if AtomicDataDict.PER_ATOM_ENERGY_KEY in model_out:
        results["energies"] = (
            model_out[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            .detach()
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    if AtomicDataDict.FORCE_KEY in model_out:
        results["forces"] = model_out[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
    return results


class AllegroCalculator(Calculator):
    """
    ASE calculator for the Allegro machine learning interatomic potential.
    
    Parameters
    ----------
    atoms : Atoms
        ASE atoms object.
    layer_symbols : list[str]
        List of symbols representing different layers in the structure.
    model_file : str
        Path to the file containing the trained model.
    device : str, optional
        Device to run the calculations on, default is 'cpu'.
    kwargs : dict
        Additional keyword arguments for the base class.
    
    Attributes
    ----------
    atoms : Atoms
        ASE atoms object.
    atom_types : array
        Array of atom types extracted from atoms object.
    layer_symbols : list[str]
        Flattened list of layer symbols.
    model : object
        Loaded trained model.
    metadata_dict : dict
        Metadata associated with the model.
    relative_layer_types : array
        Array mapping atom types to their relative positions.
    """
    # Define the properties that the calculator can handle
    implemented_properties = ["energy", "energies", "forces", "free_energy"]

    def __init__(self,
                 atoms,
                 layer_symbols: list[str],
                 model_file: str,
                 device='cpu',
                 **kwargs):
        
        self.atoms = atoms  # ASE atoms object
        self.atom_types = atoms.arrays['atom_types']  # Extract atom types from atoms object
        self.device = device  # Device for computations

        # Flatten the layer symbols list
        self.layer_symbols = [symbol for sublist in layer_symbols for symbol in (sublist if isinstance(sublist, list) else [sublist])]

        # Load the trained model and metadata
        self.model, self.metadata_dict = load_deployed_model(model_path=model_file, device=device)
        
        # Determine unique atom types and their indices
        unique_types, inverse = np.unique(self.atom_types, return_inverse=True)

        # Map atom types to their relative positions in the unique_types array
        self.relative_layer_types = inverse

        # Ensure the number of unique atom types matches the number of layer symbols provided
        if len(unique_types) != len(self.layer_symbols):
            raise ValueError("Mismatch between the number of atom types and provided layer symbols.")

        # Initialize the base Calculator class with any additional keyword arguments
        Calculator.__init__(self, **kwargs)

    def calculate(self,
                  atoms,
                  properties=None,
                  system_changes=all_changes):
        """
        Performs the calculation for the given atoms and properties.

        Parameters
        ----------
        atoms : Atoms
            ASE atoms object to calculate properties for.
        properties : list, optional
            List of properties to calculate. If None, uses implemented_properties.
        system_changes : list, optional
            List of changes that have been made to the system since last calculation.
        """
        # Default to implemented properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Create a temporary copy of the atoms object
        tmp_atoms = atoms.copy()[:]
        tmp_atoms.calc = None  # Remove any attached calculator

        r_max = self.metadata_dict["r_max"]  # Maximum radius for calculations

        # Backup original atomic numbers and set new atomic numbers based on relative layer types
        original_atom_numbers = tmp_atoms.numbers.copy()
        tmp_atoms.set_atomic_numbers(self.relative_layer_types + 1)
        tmp_atoms.arrays['atom_types'] = self.relative_layer_types

        # Prepare atomic data for the model
        data = AtomicData.from_ase(atoms=tmp_atoms, r_max=r_max, include_keys=[AtomicDataDict.ATOM_TYPE_KEY])

        # Remove energy keys from the data if present
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]

        # Move data to the specified device and convert to AtomicDataDict format
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)
        # Pass data through the model to get the output
        out = self.model(data)

        # Restore the original atomic numbers and types
        tmp_atoms.set_atomic_numbers(original_atom_numbers)
        tmp_atoms.arrays['atom_types'] = self.atom_types
        
        # Process the model output to get the desired results
        self.results = get_results_from_model_out(out)