import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import cKDTree
import os
import matplotlib.font_manager as fm    

# Set up fonts
home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:
    fm.fontManager.addfont(fnt)
    
plt.style.use("matplotlib.rc")

# Global plot settings
XLIM = (-1, 24)  # Converted from Å to nm
YLIM = (-16, 2)  # Converted from Å to nm
MARKER_SIZE = 1
MARKER_STYLE = 'H'
COLORMAP_DISREGISTRY = 'YlGn'
FIG_SIZE = (2.5, 3.5)

def get_mag_displacement(atoms):
    # Select bottom and top layer atoms
    bottom_atoms = atoms[atoms.arrays['atom_types'] == 0]
    top_atoms = atoms[atoms.arrays['atom_types'] == 3]

    # Get the 2D cell vectors
    cell_2d = atoms.cell[:2, :2]

    # Create a 3x3 periodic repetition of bottom layer atoms
    bottom_positions = bottom_atoms.positions[:, :2]
    bottom_repeated = np.vstack([bottom_positions + np.dot([i, j], cell_2d) 
                                for i in range(-1, 2) for j in range(-1, 2)])

    # Create a KD-tree for efficient nearest neighbor search
    tree = cKDTree(bottom_repeated)

    # Calculate displacement vectors for top layer atoms
    top_positions = top_atoms.positions[:, :2]
    _, indices = tree.query(top_positions)
    displacement_vectors = top_positions - bottom_repeated[indices]

    # Calculate the magnitude of displacement vectors
    displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)

    # Create periodic repetitions of top layer atoms and their displacement magnitudes
    top_repeated = np.vstack([top_positions + np.dot([i, j], cell_2d) 
                            for i in range(-1, 2) for j in range(-1, 2)])
    magnitudes_repeated = np.tile(displacement_magnitudes, 9)
    
    # Convert positions to nm
    return top_repeated / 10, magnitudes_repeated / 10

def add_cell_boundaries(ax, atoms):
    cell_2d = atoms.cell[:2, :2] / 10  # Convert to nm
    origin = (0, 0)
    corners = [origin,
               origin + cell_2d[0],
               origin + cell_2d[0] + cell_2d[1],
               origin + cell_2d[1],
               origin]
    xs, ys = zip(*corners)
    ax.plot(xs, ys, color='red', linewidth=2)

def main():
    # Read the structures
    atoms_skew_1 = read('relax_new_2mr6_cfg3_r6_N6.xyz', index=-1, format='extxyz')
    atoms_skew_2 = read('relax_new_2mr6_r6_N10.xyz', index=-1, format='extxyz')

    # Read atom types
    atom_types = read("giant_struc_2.xyz", format='extxyz', index='-1').arrays['atom_types']

    # Assign atom types to structures
    for atoms in [atoms_skew_1, atoms_skew_2]:
        atoms.arrays['atom_types'] = atom_types

    # Create figure with two vertically stacked subplots
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], 
                         top=.99,
                         bottom=0.27,
                         left=0.15,
                         right=1,
                         hspace=0.03)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    # Plot skew_1
    top_repeated_1, magnitudes_1 = get_mag_displacement(atoms_skew_1)
    scatter1 = ax1.scatter(top_repeated_1[:, 0], top_repeated_1[:, 1], 
                          marker=MARKER_STYLE, s=MARKER_SIZE,
                          c=magnitudes_1, cmap=COLORMAP_DISREGISTRY)
    ax1.set_ylabel('y (nm)')
    add_cell_boundaries(ax1, atoms_skew_1)
    ax1.axis('scaled')
    ax1.set_xlim(*XLIM)
    ax1.set_ylim(*YLIM)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', length=0)
    
    # Plot skew_2
    top_repeated_2, magnitudes_2 = get_mag_displacement(atoms_skew_2)
    scatter2 = ax2.scatter(top_repeated_2[:, 0], top_repeated_2[:, 1], 
                          marker=MARKER_STYLE, s=MARKER_SIZE,
                          c=magnitudes_2, cmap=COLORMAP_DISREGISTRY)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    add_cell_boundaries(ax2, atoms_skew_2)
    ax2.axis('scaled')
    ax2.set_xlim(*XLIM)
    ax2.set_ylim(*YLIM)
    
    # Add a single colorbar at the bottom
    cbar_ax = fig.add_axes([0.23, 0.12, 0.69, 0.02])
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation='horizontal',
                       label='Displacement (nm)', 
                       ticks=[0, 0.05, 0.10, 0.15])  # Converted from Å to nm
    cbar.ax.set_xlabel('Displacement (nm)', labelpad=2)
    
    # Save the figure
    os.makedirs('disregistry_magnitude', exist_ok=True)
    plt.savefig('disregistry_magnitude/disregistry_magnitude_comparison.pdf', 
                dpi=600, transparent=True)
    plt.close()

if __name__ == "__main__":
    main() 