import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from scipy.spatial import cKDTree
import os

plt.style.use("../matplotlib.rc")

XLIM = (-1, 24)   # nm
YLIM = (-16, 2)   # nm
MARKER_SIZE = 1
MARKER_STYLE = 'H'
COLORMAP_DISREGISTRY = 'YlGn'
FIG_SIZE = (2.5, 3.5)

def get_mag_displacement(atoms):
    bottom_atoms = atoms[atoms.arrays['atom_types'] == 0]
    top_atoms = atoms[atoms.arrays['atom_types'] == 3]

    cell_2d = atoms.cell[:2, :2]
    bottom_positions = bottom_atoms.positions[:, :2]
    bottom_repeated = np.vstack([bottom_positions + np.dot([i, j], cell_2d) 
                                for i in range(-1, 2) for j in range(-1, 2)])

    tree = cKDTree(bottom_repeated)
    top_positions = top_atoms.positions[:, :2]
    _, indices = tree.query(top_positions)
    displacement_vectors = top_positions - bottom_repeated[indices]
    displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)

    top_repeated = np.vstack([top_positions + np.dot([i, j], cell_2d) 
                            for i in range(-1, 2) for j in range(-1, 2)])
    magnitudes_repeated = np.tile(displacement_magnitudes, 9)
    
    return top_repeated / 10, magnitudes_repeated / 10  # convert Å -> nm

def add_cell_boundaries(ax, atoms):
    cell_2d = atoms.cell[:2, :2] / 10  # convert to nm
    origin = (0, 0)
    corners = [origin,
               origin + cell_2d[0],
               origin + cell_2d[0] + cell_2d[1],
               origin + cell_2d[1],
               origin]
    xs, ys = zip(*corners)
    ax.plot(xs, ys, color='red', linewidth=2)

def main():
    atom_types = read("MoS2-WSe2_1p1deg_reference.xyz", format='extxyz', index='-1').arrays['atom_types']

    atoms_um1 = read('MoS2-WSe2_1p1deg_UM1_relaxed.xyz', index=-1, format='extxyz')
    atoms_um2 = read('MoS2-WSe2_1p1deg_UM2_relaxed.xyz', index=-1, format='extxyz')

    for atoms in [atoms_um1, atoms_um2]:
        atoms.arrays['atom_types'] = atom_types

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], 
                         top=.99, bottom=0.27, left=0.15, right=1, hspace=0.03)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    top_repeated_1, magnitudes_1 = get_mag_displacement(atoms_um1)
    scatter1 = ax1.scatter(top_repeated_1[:, 0], top_repeated_1[:, 1], 
                          marker=MARKER_STYLE, s=MARKER_SIZE,
                          c=magnitudes_1, cmap=COLORMAP_DISREGISTRY)
    ax1.set_ylabel('y (nm)')
    add_cell_boundaries(ax1, atoms_um1)
    ax1.axis('scaled')
    ax1.set_xlim(*XLIM)
    ax1.set_ylim(*YLIM)
    ax1.set_xticklabels([])
    ax1.tick_params(axis='x', length=0)
    
    top_repeated_2, magnitudes_2 = get_mag_displacement(atoms_um2)
    scatter2 = ax2.scatter(top_repeated_2[:, 0], top_repeated_2[:, 1], 
                          marker=MARKER_STYLE, s=MARKER_SIZE,
                          c=magnitudes_2, cmap=COLORMAP_DISREGISTRY)
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('y (nm)')
    add_cell_boundaries(ax2, atoms_um2)
    ax2.axis('scaled')
    ax2.set_xlim(*XLIM)
    ax2.set_ylim(*YLIM)
    
    cbar_ax = fig.add_axes([0.23, 0.12, 0.69, 0.02])
    cbar = fig.colorbar(scatter1, cax=cbar_ax, orientation='horizontal',
                       label='Displacement (nm)', 
                       ticks=[0, 0.05, 0.10, 0.15])
    cbar.ax.set_xlabel('Displacement (nm)', labelpad=2)
    
    plt.savefig('fig3d_disregistry_magnitude.pdf', dpi=600, transparent=True)
    plt.close()
    print("Figure saved as 'fig3d_disregistry_magnitude.pdf'")

if __name__ == "__main__":
    main()
