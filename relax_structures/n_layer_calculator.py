from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import numpy as np

class NLayerCalculator(Calculator):
    """
    A calculator designed to handle calculations for materials with multiple layers.
    
    This calculator manages both intralayer interactions (within individual layers) 
    and interlayer interactions (between different layers) by combining separate 
    calculators for each type of interaction.
    
    Parameters
    ----------
    atoms : Atoms
        The ASE Atoms object representing the multilayer system.
    intralayer_calculators : list[Calculator]
        List of calculators for computing interactions within each layer.
        The list length should match the number of layers in the system.
    interlayer_calculators : list[Calculator]
        List of calculators for computing interactions between adjacent layers.
        The list length should be one less than the number of layers.
    layer_symbols : list[str]
        List of identifiers for each layer in the system. Used to match atoms
        to their respective layers based on atom_types.
    kwargs : dict
        Additional keyword arguments for the base Calculator class.
    
    Attributes
    ----------
    intra_calc_list : list[Calculator]
        List of calculators for intralayer interactions.
    inter_calc_list : list[Calculator]
        List of calculators for interlayer interactions.
    layer_symbols : list[str]
        Identifiers for the layers in the system.
    atoms_layer_list : list[Atoms]
        List of Atoms objects representing each separate layer after separation.
    """
    # Properties that the calculator can compute
    implemented_properties = ['energy', 'energies', 'forces']
    
    def __init__(self, atoms, intralayer_calculators: list[Calculator], interlayer_calculators: list[Calculator], layer_symbols: list[str], **kwargs):

        super().__init__(**kwargs)  # Initialize the parent Calculator class
        self.intra_calc_list = intralayer_calculators  # List of intra-layer calculators
        self.inter_calc_list = interlayer_calculators  # List of inter-layer calculators
        self.layer_symbols = layer_symbols  # Symbols for different layers

    def calculate(self, atoms: Atoms, properties=None, system_changes=all_changes):
        """
        Perform the calculation for the specified multilayer system.
        
        This method coordinates calculations for both intralayer and interlayer
        interactions, combining results into a complete energy and forces calculation.
        
        Parameters
        ----------
        atoms : Atoms
            The ASE Atoms object to calculate properties for.
        properties : list, optional
            The list of properties to calculate. Uses implemented_properties by default.
        system_changes : list, optional
            The list of changes that have been made to the system since the last calculation.
        """
        # Set to default properties if none are specified
        if properties is None:
            properties = self.implemented_properties

        # Initialize or reset the layer list for atoms
        self.atoms_layer_list = []

        # Perform the base class calculation (primarily for handling system changes)
        super().calculate(atoms, properties, system_changes)

        # Separate atoms into layers based on atom_types
        self.get_atoms_layer_list(atoms)

        # Initialize result arrays for energies and forces
        num_layers = len(self.intra_calc_list)
        num_atoms = atoms.get_global_number_of_atoms()
        self.results['layer_energy'] = np.zeros((num_layers, num_layers))
        self.results['layer_forces'] = np.zeros((num_layers, num_layers, num_atoms, 3))

        # Calculate all layer interactions (both intra and inter)
        for i in range(num_layers):
            for j in range(num_layers):    
                if i == j:
                    # Intra-layer calculation
                    self.calculate_intralayer(atoms, layer=i)
                elif i < j:
                    # Inter-layer calculation for layers i and j
                    self.calculate_interlayer(atoms, layer_1=i, layer_2=j)
                else:
                    # No calculation needed, ensure energy and forces are set to zero
                    self.results['layer_energy'][i, j] = 0
                    self.results['layer_forces'][i, j] = 0

        # Aggregate the energies and forces from all layers to the top-level results
        self.results["energy"] = self.results["layer_energy"].sum()
        self.results["forces"] = self.results["layer_forces"].sum(axis=(0, 1))

    def calculate_intralayer(self, atoms, layer: int):
        """
        Calculate properties for interactions within a specified layer.
        
        Parameters
        ----------
        atoms : Atoms
            The complete ASE Atoms object containing all layers.
        layer : int
            The index of the layer for which to calculate intralayer interactions.
            
        Raises
        ------
        ValueError
            If the specified layer index is out of range.
        """
        if layer < len(self.intra_calc_list):
            # Assign the corresponding calculator and atoms for the layer
            calc = self.intra_calc_list[layer]
            atoms_L = self.atoms_layer_list[layer]
        else:
            raise ValueError("Invalid layer index for intralayer calculation.")
        
        # Perform the calculation using the assigned intra-layer calculator
        atoms_L.calc = calc
        calc.calculate(atoms_L)

        # Update the results with energies and forces computed for this layer
        lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:layer]])
        layer_atom_indices = [lower_layers_num_atoms, lower_layers_num_atoms + len(atoms_L)]
        self.results['layer_energy'][layer, layer] = atoms_L.calc.results['energy']
        self.results['layer_forces'][layer, layer][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']

    def calculate_interlayer(self, atoms, layer_1: int, layer_2: int):
        """
        Calculate properties for interactions between two specified layers.
        
        Parameters
        ----------
        atoms : Atoms
            The complete ASE Atoms object containing all layers.
        layer_1 : int
            The index of the first layer (should be less than layer_2).
        layer_2 : int
            The index of the second layer (should be greater than layer_1).
        """
        if layer_2 <= layer_1 + 1:
            # For adjacent layers, combine atoms and calculate interlayer interactions
            atoms_L = self.atoms_layer_list[layer_1].copy() + self.atoms_layer_list[layer_2].copy()
            calc = self.inter_calc_list[layer_1]
            atoms_L.calc = calc
            calc.calculate(atoms_L)
            
            # Update the results with energies and forces computed between these layers
            lower_layers_num_atoms = sum([len(layer_atoms) for layer_atoms in self.atoms_layer_list[:layer_1]])
            layer_atom_indices = [lower_layers_num_atoms, lower_layers_num_atoms + len(self.atoms_layer_list[layer_1]) + len(self.atoms_layer_list[layer_2])]
            self.results['layer_energy'][layer_1, layer_2] = atoms_L.calc.results['energy']
            self.results['layer_forces'][layer_1, layer_2][layer_atom_indices[0]:layer_atom_indices[1]] = atoms_L.calc.results['forces']

    def get_atoms_layer_list(self, atoms):
        """
        Separate atoms into different layers based on atom_types.
        
        This method populates the atoms_layer_list attribute with separate
        Atoms objects for each layer in the system.
        
        Parameters
        ----------
        atoms : Atoms
            The complete ASE Atoms object to be separated into layers.
        """
        for layer in range(len(self.intra_calc_list)):
            # Determine the range of atom types for the current layer
            lower_layer_num_atoms = sum([len(layer_atoms) for layer_atoms in self.layer_symbols[:layer]])
            upper_layer_num_atoms = lower_layer_num_atoms + len(self.layer_symbols[layer]) - 1
            
            # Select atoms within the specified range of atom types
            atoms_L = atoms.copy()[np.logical_and(atoms.arrays["atom_types"] <= upper_layer_num_atoms,
                                                  atoms.arrays["atom_types"] >= lower_layer_num_atoms)]
            
            # Add the selected atoms to the layer list
            self.atoms_layer_list.append(atoms_L)
            
