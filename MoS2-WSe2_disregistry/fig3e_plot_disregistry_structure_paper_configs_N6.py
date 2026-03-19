import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree, Delaunay
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
import matplotlib.animation as animation
import os
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import time  # Import time module for timing

import matplotlib.font_manager as fm    
home_dir = os.environ["HOME"]
fnt_pths = os.path.join(home_dir, "fonts")
fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
for fnt in fnts_files:

    fm.fontManager.addfont(fnt)
    
plt.style.use("matplotlib.rc")

# Global settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

n_bins = 256
atom_index_cmap = mcolors.LinearSegmentedColormap.from_list('cyclic', [
    (0.0, '#d62728'),  # Red at 0
    (1/3, '#1f77b4'),  # Blue at 1/3
    (0.5, '#ff7f0e'),  # Orange at 1/2
    (2/3, '#2ca02c'),  # Green at 2/3
    (1.0, '#d62728')   # Red at 1
], N=n_bins)

displacement_magnitude_cmap = plt.cm.YlGn  # Changed from viridis to YlGn

def unit_range_fixed(x, L=1, eps=1e-9):
    y = x.copy() % L
    y[(np.fabs(y) < eps) | (np.fabs(L - y) < eps)] = 0
    return y

def convert_string_to_array(string):
    return np.array([float(x) for x in string.strip('[]').split()]).reshape(2, 2)

def pad_periodic_image(pos, box, n_a1=1, n_a2=1):
    print("Padding periodic image")
    i_range = np.concatenate((np.arange(0, n_a1 + 1), np.arange(-n_a1, 0)))
    j_range = np.concatenate((np.arange(0, n_a2 + 1), np.arange(-n_a2, 0)))
    i, j = np.meshgrid(i_range, j_range)
    offsets = i.flatten()[:, np.newaxis] * box[0] + j.flatten()[:, np.newaxis] * box[1]
    padded_pos = pos[np.newaxis, :, :] + offsets[:, np.newaxis, :]
    return np.vstack((pos, padded_pos[1:].reshape(-1, 2)))

def plot_voronoi_diagram(vor, xlim, ylim, color='k', label=None, lw=0.5, a=1.0):
    ridge_vertices = np.array(vor.ridge_vertices)
    valid_ridges = ridge_vertices[(ridge_vertices >= 0).all(axis=1)]
    vertices = vor.vertices[valid_ridges]
    within_limits = np.all((vertices[:, :, 0] >= xlim[0]) & (vertices[:, :, 0] <= xlim[1]) &
                           (vertices[:, :, 1] >= ylim[0]) & (vertices[:, :, 1] <= ylim[1]), axis=1)
    lines = vertices[within_limits]
    return LineCollection(lines, colors=color, linewidths=lw, alpha=a, label=label)

def get_primitive_voronoi_cell(A1):
    x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    points = np.column_stack((x.ravel(), y.ravel()))
    lattice_points = points @ A1
    vor = Voronoi(lattice_points)
    central_region = vor.regions[vor.point_region[4]]
    return vor.vertices[central_region] if -1 not in central_region else None

def voronoi_interpolation(relaxed_points, unrelaxed_points, 
                          query_points, pristine_voronoi_vertices):
    print(f"\nStarting voronoi_interpolation with:")
    print(f"- {len(relaxed_points)} relaxed points")
    print(f"- {len(unrelaxed_points)} unrelaxed points")
    print(f"- {len(query_points)} query points")
    print(f"- {len(pristine_voronoi_vertices)} pristine voronoi vertices")

    def find_voronoi_cell(query_point, points, tri, circumcenters):
        distances = np.sum((points - query_point)**2, axis=1)
        nearest_point_index = np.argmin(distances)
        print(f"\nFound nearest point index: {nearest_point_index}")
        
        simplices_containing_point = np.where((tri.simplices == nearest_point_index).any(axis=1))[0]
        print(f"Number of simplices containing point: {len(simplices_containing_point)}")
        
        cell_indices = simplices_containing_point[np.argsort(np.arctan2(
            circumcenters[simplices_containing_point][:, 1] - points[nearest_point_index, 1],
            circumcenters[simplices_containing_point][:, 0] - points[nearest_point_index, 0]
        ))]
        return cell_indices, nearest_point_index

    def calculate_circumcenter(triangle):
        a, b, c = triangle
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-10:  # Check if d is very close to zero
            print("Warning: Denominator close to zero in circumcenter calculation, using fallback")
            # Return the midpoint of the longest side as a fallback
            sides = [np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(a-c)]
            longest_side_index = np.argmax(sides)
            if longest_side_index == 0:
                return (a + b) / 2
            elif longest_side_index == 1:
                return (b + c) / 2
            else:
                return (c + a) / 2
        ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
        uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
        return np.array([ux, uy])

    def map_to_pristine_voronoi(points, original_vertices, pristine_vertices, original_center, pristine_center):
        print(f"\nMapping to pristine Voronoi cell:")
        print(f"- Original vertices shape: {original_vertices.shape}")
        print(f"- Pristine vertices shape: {pristine_vertices.shape}")
        
        original_vertices_centered = original_vertices - (original_center - pristine_center)
        pristine_vertices_centered = pristine_vertices - pristine_center
        cost_matrix = cdist(original_vertices_centered, pristine_vertices_centered)
        _, col_ind = linear_sum_assignment(cost_matrix)
        print(f"- Assignment cost: {cost_matrix[np.arange(len(col_ind)), col_ind].sum():.6f}")
        
        pristine_vertices_matched = pristine_vertices_centered[col_ind]
        tps = RBFInterpolator(original_vertices_centered, pristine_vertices_matched, kernel='thin_plate_spline', smoothing=0)
        points_centered = points - (original_center - pristine_center)
        transformed_points = tps(points_centered)
        mapped_points = transformed_points + pristine_center
        return mapped_points

    print("\nCreating Delaunay triangulation...")
    tri_relaxed = Delaunay(relaxed_points)
    print(f"- Number of relaxed simplices: {len(tri_relaxed.simplices)}")
    
    print("\nCalculating circumcenters...")
    circumcenters_relaxed = np.array([calculate_circumcenter(relaxed_points[simplex]) for simplex in tri_relaxed.simplices])
    
    tri_unrelaxed = Delaunay(unrelaxed_points)
    print(f"- Number of unrelaxed simplices: {len(tri_unrelaxed.simplices)}")
    circumcenters_unrelaxed = np.array([calculate_circumcenter(unrelaxed_points[simplex]) for simplex in tri_unrelaxed.simplices])

    print("\nCreating RBF interpolator...")
    rbf_interpolator = RBFInterpolator(unrelaxed_points, relaxed_points, kernel='thin_plate_spline')
    circumcenters_rbf = rbf_interpolator(circumcenters_unrelaxed)

    pristine_center = np.mean(pristine_voronoi_vertices, axis=0)
    pristine_voronoi_vertices_cell = np.vstack([pristine_voronoi_vertices, pristine_center])
    interpolated_points_pristine = []
    interpolated_points_unrelaxed = []

    print("\nProcessing query points...")
    for i, query_point in enumerate(query_points):
        print(f"\nProcessing query point {i+1}/{len(query_points)}")
        voronoi_indices, nearest_point_index = find_voronoi_cell(query_point, relaxed_points, tri_unrelaxed, circumcenters_rbf)
        voronoi_vertices_relaxed = np.vstack([relaxed_points[nearest_point_index], circumcenters_rbf[voronoi_indices]])
        voronoi_vertices_unrelaxed = np.vstack([unrelaxed_points[nearest_point_index], circumcenters_unrelaxed[voronoi_indices]])
        
        print(f"Creating RBF interpolator for Voronoi cell with {len(voronoi_vertices_relaxed)} vertices")
        rbf_voronoi = RBFInterpolator(voronoi_vertices_relaxed, voronoi_vertices_unrelaxed)
        interpolated_point = rbf_voronoi(query_point.reshape(1, -1))[0]

        # Map to pristine cell
        mapped_point_pristine = map_to_pristine_voronoi(
            interpolated_point.reshape(1, -1),
            voronoi_vertices_unrelaxed,
            pristine_voronoi_vertices_cell,
            unrelaxed_points[nearest_point_index],
            pristine_center
        )[0]
        
        interpolated_points_pristine.append(mapped_point_pristine)
        interpolated_points_unrelaxed.append(interpolated_point)  # Changed to store interpolated point directly

    print("\nInterpolation complete!")
    return (np.array(interpolated_points_pristine), 
            np.array(interpolated_points_unrelaxed), 
            circumcenters_unrelaxed, 
            circumcenters_rbf)


def create_single_displacement_plot(displacements_list, A1, confined_displacements_energy, energies, 
                                  output_file='single_displacement_plot_color.png', 
                                  use_magnitude_colors=False):
    print("Creating single displacement plot")
    pristine_cell = get_primitive_voronoi_cell(A1)
    centroid = np.mean(pristine_cell, axis=0)
    energies = energies * 1000
    
    x = np.linspace(centroid[0] - 1.5*np.linalg.norm(A1[0]), centroid[0] + 1.5*np.linalg.norm(A1[0]), 200)
    y = np.linspace(centroid[1] - 1.5*np.linalg.norm(A1[1]), centroid[1] + 1.5*np.linalg.norm(A1[1]), 200)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    print("Plotting Energy")
    n_repeats = 1
    repeated_displacements = []
    repeated_energies = []
    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            repeated_displacements.append(confined_displacements_energy + offset)
            repeated_energies.append(energies)
    
    repeated_displacements = np.vstack(repeated_displacements)
    repeated_energies = np.concatenate(repeated_energies)

    rbf = RBFInterpolator(repeated_displacements + centroid, repeated_energies, kernel='thin_plate_spline', smoothing=0.1)
    interpolated_energies = rbf(grid_points).reshape(X.shape)
    plt.style.use("./matplotlib.rc")
    fig, ax = plt.subplots(figsize =(1.25,1.25), dpi=300,layout='constrained')
    atom_norm = plt.Normalize(0, len(displacements_list)-1)
    extent = [x.min(), x.max(), y.min(), y.max()]
    
    energy_norm = mcolors.Normalize(vmin=np.min(energies), vmax=np.max(energies))
    
    im = ax.imshow(interpolated_energies, extent=extent, origin='lower', cmap='gist_gray', 
                   norm=energy_norm, aspect='equal', alpha=0.7)
    
    # cbar_energy = plt.colorbar(im, ax=ax, shrink=1.0, pad=0.02)
    # cbar_energy.ax.tick_params(direction='out')
    # cbar_energy.set_label('Energy Relative to AA Stacking (meV)')

    print("Plotting Voronoi")
    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            cell = pristine_cell + offset
            ax.plot(np.append(cell[:, 0], cell[0, 0]),
                    np.append(cell[:, 1], cell[0, 1]), 'lightskyblue', linewidth=1, alpha=0.4, zorder=6,
                    label='Mo Vornoi' if i == 0 and j == 0 else None)
            cell_center = np.mean(cell, axis=0)
            # ax.plot(cell_center[0], cell_center[1], 'bx', markersize=10, markeredgewidth=2, zorder=2)
    
    print("Plotting Displacements")
    
    n_repeats = 1
    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            for idx, displacement in enumerate(displacements_list):
                end_point = centroid + displacement + offset
                
                if use_magnitude_colors:
                    # Calculate displacement magnitude and normalize it
                    magnitude = np.linalg.norm(displacement)
                    max_magnitude = np.max([np.linalg.norm(d) for d in displacements_list])
                    normalized_magnitude = magnitude / max_magnitude
                    color = displacement_magnitude_cmap(normalized_magnitude)
                    ax.scatter(end_point[0], end_point[1], 
                              color=color, s=2, alpha=1.0, marker='h',
                              zorder=5)
                else:
                    ax.scatter(end_point[0], end_point[1], 
                              c=[[idx]], s=2, alpha=1.0, marker='h',
                              cmap=atom_index_cmap, norm=atom_norm,
                              zorder=5)

    padding_factor = .6
    ax.set_xlim(centroid[0] - (padding_factor)*np.linalg.norm(A1[0]), centroid[0] + (padding_factor)*np.linalg.norm(A1[0]))
    ax.set_ylim(centroid[1] - (padding_factor)*np.linalg.norm(A1[0]), centroid[1] + (padding_factor)*np.linalg.norm(A1[1]))
    ax.set_aspect('equal')
    # ax.set_xlabel(r'$\delta \mathbf{r}_x$ (Å)')
    # ax.set_ylabel(r'$\delta \mathbf{r}_y$ (Å)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)  # Remove gridlines

    # if use_magnitude_colors:
    #     max_magnitude = np.max([np.linalg.norm(d) for d in displacements_list])
    #     norm = plt.Normalize(0, max_magnitude)
    #     sm = plt.cm.ScalarMappable(cmap=displacement_magnitude_cmap, norm=norm)
    #     cbar = plt.colorbar(sm, ax=ax)
    #     cbar.set_label('Displacement Magnitude (Å)')

    # Create the directory if it doesn't exist
    output_dir = 'disregistry_plots'
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, output_file), dpi=300, transparent=True)
    plt.close(fig)
    print(f"Single displacement plot saved as '{os.path.join(output_dir, output_file)}' with transparent background")

def main(structure_file, energy_file, output_file, displacements_file='displacements_list.npy', 
         use_magnitude_colors=False):
    print("Starting disregistry analysis...")
    start_time = time.time()  # Start timing

    if not all(os.path.isfile(f) for f in [structure_file, energy_file]):
        print("Error: One or more input files do not exist.")
        return

    # Timing reading structure file
    start_read_structure = time.time()
    print(f"Reading structure file: {structure_file}")
    structures = read(structure_file, index='-1', format='extxyz')
    reference_structure = read("giant_struc_2.xyz", index='-1', format='extxyz')
    A1 = convert_string_to_array(reference_structure.info['base_lattice_0'])
    structures.arrays['atom_types'] = reference_structure.arrays['atom_types']
    structures = [structures]
    pristine_cell = get_primitive_voronoi_cell(A1)
    print(f"Primitive Voronoi cell vertices: {pristine_cell}")
    end_read_structure = time.time()

    # Timing reading energy file
    start_read_energy = time.time()
    print(f"Reading energy file: {energy_file}")
    energy_structures = read(energy_file, index=':', format='extxyz')
    end_read_energy = time.time()

    # Timing displacement calculation
    start_displacement_calc = time.time()
    print("Calculating displacements for energy structures...")
    confined_displacements_list, energies = [], []

    vor_energy, centroids_energy, displacements_energy, confined_displacements_energy, energies = [], [], [], [], []

    for structure in energy_structures:
        padded_energy_pos = pad_periodic_image(structure.positions[structure.arrays['atom_types'] == 0, :2], structure.cell[:2, :2], n_a1=3, n_a2=3)
        query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]
    
        interpolated_points_pristine, _, _, _ = voronoi_interpolation(padded_energy_pos,
                                                            padded_energy_pos,
                                                            query_points,
                                                            pristine_cell)

        confined_displacements_energy.append(interpolated_points_pristine)
        energies.append(structure.get_potential_energy())
        

    confined_displacements_energy = np.array(confined_displacements_energy).reshape(-1, 2)
    energies = np.array(energies)
    
    max_energy = np.max(energies)
    energies -= max_energy

    print(f"Processed {len(energy_structures)} energy structures.")
    print(f"Shifted energies by maximum energy: {max_energy:.6f} eV")
    end_displacement_calc = time.time()

    # Timing displacements list calculation
    start_displacements_list = time.time()
    if os.path.isfile(displacements_file):
        print(f"Displacements file '{displacements_file}' already exists. Loading displacements list from file.")
        displacements_list = np.load(displacements_file)
    else:
        displacements_list = []
        for structure in structures:
            padded_pos = pad_periodic_image(structure.positions[structure.arrays['atom_types'] == 0, :2], structure.cell[:2, :2])
            query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]

            interpolated_points_pristine, _, _, _ = voronoi_interpolation(padded_pos,
                                                                  padded_pos,
                                                                  query_points,
                                                                  pristine_cell)
            displacements_list.append(interpolated_points_pristine)
        displacements_list = np.array(displacements_list).reshape(-1, 2)
        
        print(f"Displacements list calculated for {len(structures)} structures.")
        np.save(displacements_file, displacements_list)
        print(f"Displacements list saved to '{displacements_file}'")
    end_displacements_list = time.time()

    # Timing plot creation
    start_plot_creation = time.time()
    print("Generating single displacement plot...")
    create_single_displacement_plot(displacements_list, A1, confined_displacements_energy, energies, 
                                  output_file, use_magnitude_colors)
    end_plot_creation = time.time()
    
    print("Disregistry analysis completed.")

    # Print timing summary
    print("\nTiming Summary:")
    print(f"Reading structure file: {end_read_structure - start_read_structure:.2f} seconds")
    print(f"Reading energy file: {end_read_energy - start_read_energy:.2f} seconds")
    print(f"Displacement calculation: {end_displacement_calc - start_displacement_calc:.2f} seconds")
    print(f"Displacements list calculation: {end_displacements_list - start_displacements_list:.2f} seconds")
    print(f"Plot creation: {end_plot_creation - start_plot_creation:.2f} seconds")
    print(f"Total time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    print("Initializing font manager")
    import matplotlib.font_manager as fm    
    home_dir = os.environ["HOME"]
    fnt_pths = os.path.join(home_dir, "fonts")
    fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
    for fnt in fnts_files:
        fm.fontManager.addfont(fnt)

    structure_file = 'relax_new_2mr6_cfg3_r6_N6.xyz'
    energy_file = 'bilayer_MoS2_WSe2_config_min_dist_configuration_space.xyz'
    output_file = 'single_displacement_plot_color_n6.png'
    displacements_file = 'displacements_list_n6.npy'
    use_magnitude_colors = True  # Set to True to use magnitude-based coloring
    main(structure_file, energy_file, output_file, displacements_file, use_magnitude_colors)
