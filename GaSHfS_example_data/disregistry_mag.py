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
XLIM = (-2.5, 16.5)  # Converted from Å to nm
YLIM = (-18, 1)  # Converted from Å to nm
MARKER_SIZE = 3
MARKER_STYLE = 'H'
COLORMAP_DISREGISTRY = 'RdBu'
FIG_SIZE = (3, 3.1)  # Slightly reduced height

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
    n_repeat = 6
    print(n_repeat//2)
    top_repeated = np.vstack([top_positions + np.dot([i, j], cell_2d) 
                            for i in range(-n_repeat//2, n_repeat//2) for j in range(-n_repeat//2, n_repeat//2)])
    print(top_repeated.shape)
    print(top_positions.shape)

    magnitudes_repeated = np.tile(displacement_magnitudes, n_repeat**2)
    
    return top_repeated, magnitudes_repeated

def add_cell_boundaries(ax, atoms):
    cell_2d = atoms.cell[:2, :2]
    origin = (0, 0)
    corners = [origin,
               origin + cell_2d[0],
               origin + cell_2d[0] + cell_2d[1],
               origin + cell_2d[1],
               origin]
    xs, ys = zip(*corners)
    ax.plot(xs, ys, color='green', linewidth=2, alpha=0.5)

def main():
    # Read the structures
    atoms = read('relax_traj_2D_fastnl_3.xyz', index=-2, format='extxyz')
    # atoms_skew_2 = read('relax_new_2mr6_r6_N10.xyz', index=-1, format='extxyz')

    # Create figure with two vertically stacked subplots
    fig = plt.figure(figsize=FIG_SIZE)  # Remove layout='constrained'

    # Create main axes with adjusted position [left, bottom, width, height]
    ax1 = fig.add_axes([0.2, 0.35, 0.65, 0.65])  # Adjusted to make room for colorbar at bottom
    
    # Plot atoms
    top_repeated_1, magnitudes_1 = get_mag_displacement(atoms)
    scatter1 = ax1.scatter(top_repeated_1[:, 0]/10, top_repeated_1[:, 1]/10,  # Convert to nm
                          marker=MARKER_STYLE, s=MARKER_SIZE,
                          c=magnitudes_1, cmap=COLORMAP_DISREGISTRY,
                          norm=plt.Normalize(0, 1.85))
    ax1.set_ylabel('y (nm)')
    ax1.set_xlabel('x (nm)')
    # add_cell_boundaries(ax1, atoms)
    ax1.axis('scaled')
    ax1.set_xlim(*XLIM)
    ax1.set_ylim(*YLIM)
    ax1.grid(False)
    
    # Add colorbar at the bottom
    cbar_ax = fig.add_axes([0.2, 0.15, 0.65, 0.03])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation='horizontal',
                       label='Displacement Magnitude (Å)', 
                       ticks=[0, 0.6, 1.2, 1.8]) 
    
    # Save the figure
    os.makedirs('disregistry_magnitude', exist_ok=True)
    plt.savefig('disregistry_magnitude/GaSHfS_2D_disregistry_magnitude.png', 
                dpi=1200, transparent=True)
    plt.close()

if __name__ == "__main__":
    main() 