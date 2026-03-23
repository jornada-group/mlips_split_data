import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import cKDTree
import os
from matplotlib.colors import ListedColormap, Normalize

plt.style.use("../matplotlib.rc")

# Global plot settings
XLIM = (-2.5, 16.5)  # Converted from Å to nm
YLIM = (-18, 1)  # Converted from Å to nm
MARKER_SIZE = 3
MARKER_STYLE = 'H'
# Linear Normalize(vmin, vmax); custom YlGn so light/white covers more of the
# low end of the scale and dark green is compressed toward high magnitudes.
DISPLACEMENT_VMAX = 1.85
# Build colors as YlGn(p**gamma) for p in [0,1]. gamma > 1 stretches the pale
# band across lower data values and pushes greens to higher values.
YLGN_INDEX_GAMMA = 1.9
N_CMAP_COLORS = 256
FIG_SIZE = (3, 3.1)


def make_shifted_ylgn_cmap(gamma=YLGN_INDEX_GAMMA, n=N_CMAP_COLORS):
    """Resampled YlGn: more white/yellow at low magnitudes, green at high."""
    base = plt.get_cmap("YlGn")
    p = np.linspace(0.0, 1.0, n)
    colors = base(p ** float(gamma))
    return ListedColormap(colors, name="YlGn_green_at_high")


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
    top_repeated = np.vstack([top_positions + np.dot([i, j], cell_2d) 
                            for i in range(-n_repeat//2, n_repeat//2) for j in range(-n_repeat//2, n_repeat//2)])

    magnitudes_repeated = np.tile(displacement_magnitudes, n_repeat**2)
    
    return top_repeated, magnitudes_repeated


def main():
    atoms = read('GaS-HfS2_2D_moire_MLIP_relaxed.xyz', index=0, format='extxyz')

    fig = plt.figure(figsize=FIG_SIZE)

    ax1 = fig.add_axes([0.2, 0.35, 0.65, 0.65])

    top_repeated_1, magnitudes_1 = get_mag_displacement(atoms)
    norm = Normalize(vmin=0, vmax=DISPLACEMENT_VMAX)
    scatter1 = ax1.scatter(
        top_repeated_1[:, 0] / 10, top_repeated_1[:, 1] / 10,  # Convert to nm
        marker=MARKER_STYLE, s=MARKER_SIZE,
        c=magnitudes_1,
        cmap=make_shifted_ylgn_cmap(),
        norm=norm,
    )
    ax1.set_ylabel('y (nm)')
    ax1.set_xlabel('x (nm)')
    ax1.axis('scaled')
    ax1.set_xlim(*XLIM)
    ax1.set_ylim(*YLIM)
    ax1.grid(False)

    cbar_ax = fig.add_axes([0.2, 0.15, 0.65, 0.03])
    fig.colorbar(
        scatter1, cax=cbar_ax, orientation='horizontal',
        label='Displacement Magnitude (Å)',
        ticks=[0, 0.6, 1.2, 1.8],
    )

    plt.savefig('fig6d_2D_disregistry_magnitude.png', dpi=1200, transparent=True)
    plt.close()


if __name__ == "__main__":
    main()
