import time
import os
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Voronoi, cKDTree, voronoi_plot_2d, Delaunay
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import KernelDensity
import ot

np.random.seed(42)

###############################################################################
# Utility Functions
###############################################################################

def print_elapsed_time(start_time, message):
    """Print the elapsed time for a given operation."""
    elapsed_time = time.time() - start_time
    print(f"{message} - Elapsed time: {elapsed_time:.2f} seconds")
    
def convert_lattice_string_to_array(lattice_string):
    """Convert a string representation of a lattice to a 2x2 numpy array."""
    return np.array([float(x) for x in lattice_string.strip('[]').split()]).reshape(2, 2)

###############################################################################
# Voronoi Cell Functions
###############################################################################

def get_primitive_voronoi_cell(lattice_vector):
    """Generate the primitive Voronoi cell for the given lattice vector."""
    x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    points = np.column_stack((x.ravel(), y.ravel()))
    lattice_points = points @ lattice_vector
    vor = Voronoi(lattice_points)
    central_point_index = 4  # Center point in 3x3 grid
    central_region = vor.regions[vor.point_region[central_point_index]]
    if -1 not in central_region:  # Check if region is valid
        return vor.vertices[central_region]
    return None

def plot_voronoi_diagram(voronoi, centroids, step_number):
    """Plot Voronoi diagram with centroids for visualization."""
    fig, ax = plt.subplots(figsize=(10, 10))
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, 
                    line_colors='gray', line_width=1, line_alpha=0.6, point_size=2)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='blue', s=20, label='Centroids')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-2.5, 75)
    ax.set_ylim(-10, 10)
    ax.set_title(f'Voronoi Diagram and Displacements - Step {step_number}')
    plt.tight_layout()
    plt.savefig(f'voronoi_diagram_step_{step_number}.png')
    plt.close()

###############################################################################
# Displacement Calculation Functions
###############################################################################

def calculate_voronoi_and_displacement(relaxed_positions_padded, query_points, lattice_vector, num_points, step):
    """Calculate Voronoi diagrams and displacement vectors."""
    points_2d = relaxed_positions_padded[:, :2]
    voronoi = Voronoi(points_2d)
    
    primitive_cell = get_primitive_voronoi_cell(lattice_vector)
    first_n_indices = np.arange(num_points)
    first_n_regions = [voronoi.regions[voronoi.point_region[i]] for i in first_n_indices]
    centroids = points_2d[first_n_indices]
    
    # Find nearest centroids for query points
    tree = cKDTree(centroids)
    _, nearest_indices = tree.query(query_points)
    closest_centroid_points = centroids[nearest_indices]
    
    # Calculate displacements
    raw_displacements = query_points - closest_centroid_points
    confined_displacements = np.array([
        confine_displacement(disp, voronoi.vertices[region], primitive_cell) 
        for disp, region in zip(raw_displacements, first_n_regions)
    ])

    return voronoi, centroids, raw_displacements, confined_displacements

def confine_displacement(displacement, actual_cell, pristine_cell):
    """
    Confines the displacement vector within the pristine Voronoi cell using thin plate spline interpolation.
    """
    actual_centroid = np.mean(actual_cell, axis=0)
    pristine_centroid = np.mean(pristine_cell, axis=0)

    actual_cell_centered = actual_cell - actual_centroid
    pristine_cell_centered = pristine_cell - pristine_centroid
    
    cost_matrix = cdist(actual_cell_centered, pristine_cell_centered)
    _, col_ind = linear_sum_assignment(cost_matrix)

    pristine_cell_matched = pristine_cell[col_ind]

    tps = RBFInterpolator(actual_cell, pristine_cell_matched, kernel='thin_plate_spline', smoothing=0)

    start_point = actual_centroid
    end_point = start_point + displacement

    transformed_points = tps(np.vstack((start_point, end_point)))
    confined_displacement = transformed_points[1] - transformed_points[0]

    return confined_displacement

###############################################################################
# Periodic Image and Cell Functions 
###############################################################################

def pad_periodic_image(positions, lattice_vectors, n_repeat_x=1, n_repeat_y=1):
    """Create periodic images of atomic positions by repeating the unit cell.
    
    Args:
        positions: Original atomic positions
        lattice_vectors: Unit cell lattice vectors
        n_repeat_x: Number of repetitions in x direction
        n_repeat_y: Number of repetitions in y direction
    """
    x_range = np.concatenate((np.arange(0, n_repeat_x + 1), np.arange(-n_repeat_x, 0)))
    y_range = np.concatenate((np.arange(0, n_repeat_y + 1), np.arange(-n_repeat_y, 0)))
    x_indices, y_indices = np.meshgrid(x_range, y_range)
    x_indices, y_indices = x_indices.flatten(), y_indices.flatten()
    
    # Calculate offsets for each periodic image
    offsets = x_indices[:, np.newaxis] * lattice_vectors[0] + y_indices[:, np.newaxis] * lattice_vectors[1]
    padded_positions = positions[np.newaxis, :, :] + offsets[:, np.newaxis, :]
    return padded_positions

###############################################################################
# Voronoi Interpolation Functions
###############################################################################

def calculate_circumcenter(triangle_points):
    """Calculate the circumcenter of a triangle."""
    a, b, c = triangle_points
    epsilon = 1e-10  # Small value to prevent division by zero
    
    # Calculate denominator for circumcenter formula
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    
    if abs(d) < epsilon:
        # Handle near-collinear points by returning centroid
        return np.mean(triangle_points, axis=0)
        
    # Calculate circumcenter coordinates
    ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + 
          (b[0]**2 + b[1]**2) * (c[1] - a[1]) + 
          (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
    uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + 
          (b[0]**2 + b[1]**2) * (a[0] - c[0]) + 
          (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
    
    return np.array([ux, uy])

def find_voronoi_cell(query_point, points, triangulation, circumcenters):
    """Find the Voronoi cell containing a query point."""
    # Find nearest point
    distances = np.sum((points - query_point)**2, axis=1)
    nearest_point_idx = np.argmin(distances)
    
    # Find simplices containing the nearest point
    simplices_with_point = np.where((triangulation.simplices == nearest_point_idx).any(axis=1))[0]
    
    # Sort cell indices by angle around the nearest point
    angles = np.arctan2(
        circumcenters[simplices_with_point][:, 1] - points[nearest_point_idx, 1],
        circumcenters[simplices_with_point][:, 0] - points[nearest_point_idx, 0]
    )
    cell_indices = simplices_with_point[np.argsort(angles)]
    
    return cell_indices, nearest_point_idx

def map_to_pristine_voronoi(points, original_vertices, pristine_vertices, original_center, pristine_center):
    """Map points from original Voronoi cell to pristine cell."""
    # Center the vertices
    original_vertices_centered = original_vertices - (original_center - pristine_center)
    pristine_vertices_centered = pristine_vertices - pristine_center
    
    # Find optimal matching between vertices
    cost_matrix = cdist(original_vertices_centered, pristine_vertices_centered)
    _, col_indices = linear_sum_assignment(cost_matrix)
    pristine_vertices_matched = pristine_vertices_centered[col_indices]
    
    # Create interpolator and transform points
    interpolator = RBFInterpolator(original_vertices_centered, pristine_vertices_matched, 
                                 kernel='thin_plate_spline', smoothing=0)
    points_centered = points - (original_center - pristine_center)
    transformed_points = interpolator(points_centered)
    mapped_points = transformed_points + pristine_center
    
    return mapped_points

def voronoi_interpolation(relaxed_points, unrelaxed_points, 
                          query_points, pristine_voronoi_vertices):
    """
    Interpolate points from the relaxed structure to the unrelaxed structure using Voronoi method.
    
    Parameters:
    - relaxed_points: (N, 2) array of relaxed structure points
    - unrelaxed_points: (N, 2) array of unrelaxed structure points
    - query_points: (M, 2) array of points to interpolate
    - pristine_voronoi_vertices: (K, 2) array of vertices of the pristine Voronoi cell
    
    Returns:
    - interpolated_points_pristine: (M, 2) array of interpolated points within the pristine cell
    - interpolated_points_unrelaxed: (M, 2) array of interpolated points within the unrelaxed structure
    - circumcenters_unrelaxed: (L, 2) array of circumcenters in the unrelaxed structure
    - circumcenters_rbf: (L, 2) array of interpolated circumcenters in the relaxed structure
    """

    def find_voronoi_cell(query_point, points, tri, circumcenters):
        distances = np.sum((points - query_point)**2, axis=1)
        nearest_point_index = np.argmin(distances)
        simplices_containing_point = np.where((tri.simplices == nearest_point_index).any(axis=1))[0]
        cell_indices = simplices_containing_point[np.argsort(np.arctan2(
            circumcenters[simplices_containing_point][:, 1] - points[nearest_point_index, 1],
            circumcenters[simplices_containing_point][:, 0] - points[nearest_point_index, 0]
        ))]
        return cell_indices, nearest_point_index

    def calculate_circumcenter(triangle):
        a, b, c = triangle
        epsilon = 1e-10  # Small value to prevent division by zero
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < epsilon:
            # Handle the case where points are collinear or very close
            # You can return the centroid of the triangle as a fallback
            return np.mean(triangle, axis=0)
        ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) + (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
        uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) + (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
        return np.array([ux, uy])

    def map_to_pristine_voronoi(points, original_vertices, pristine_vertices, original_center, pristine_center):
        original_vertices_centered = original_vertices - (original_center - pristine_center)
        pristine_vertices_centered = pristine_vertices - pristine_center
        cost_matrix = cdist(original_vertices_centered, pristine_vertices_centered)
        _, col_ind = linear_sum_assignment(cost_matrix)
        pristine_vertices_matched = pristine_vertices_centered[col_ind]
        tps = RBFInterpolator(original_vertices_centered, pristine_vertices_matched, kernel='thin_plate_spline', smoothing=0)
        points_centered = points - (original_center - pristine_center)
        transformed_points = tps(points_centered)
        mapped_points = transformed_points + pristine_center
        return mapped_points

    tri_relaxed = Delaunay(relaxed_points)
    circumcenters_relaxed = np.array([calculate_circumcenter(relaxed_points[simplex]) for simplex in tri_relaxed.simplices])
    
    tri_unrelaxed = Delaunay(unrelaxed_points)
    circumcenters_unrelaxed = np.array([calculate_circumcenter(unrelaxed_points[simplex]) for simplex in tri_unrelaxed.simplices])

    rbf_interpolator = RBFInterpolator(unrelaxed_points, relaxed_points, kernel='thin_plate_spline')
    circumcenters_rbf = rbf_interpolator(circumcenters_unrelaxed)

    pristine_center = np.mean(pristine_voronoi_vertices, axis=0)
    pristine_voronoi_vertices_cell = np.vstack([pristine_voronoi_vertices, pristine_center])
    interpolated_points_pristine = []
    interpolated_points_unrelaxed = []

    for query_point in query_points:
        voronoi_indices, nearest_point_index = find_voronoi_cell(query_point, relaxed_points, tri_unrelaxed, circumcenters_rbf)
        voronoi_vertices_relaxed = np.vstack([relaxed_points[nearest_point_index], circumcenters_rbf[voronoi_indices]])
        voronoi_vertices_unrelaxed = np.vstack([unrelaxed_points[nearest_point_index], circumcenters_unrelaxed[voronoi_indices]])
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
        
        # Map to unrelaxed cell
        unrelaxed_cell_indices, _ = find_voronoi_cell(unrelaxed_points[nearest_point_index], unrelaxed_points, tri_unrelaxed, circumcenters_unrelaxed)
        unrelaxed_cell_vertices = np.vstack([unrelaxed_points[nearest_point_index], circumcenters_unrelaxed[unrelaxed_cell_indices]])
        mapped_point_unrelaxed = map_to_pristine_voronoi(
            interpolated_point.reshape(1, -1),
            voronoi_vertices_unrelaxed,
            unrelaxed_cell_vertices,
            unrelaxed_points[nearest_point_index],
            unrelaxed_points[nearest_point_index]
        )[0]
        
        interpolated_points_unrelaxed.append(mapped_point_unrelaxed)

    return (np.array(interpolated_points_pristine), 
            np.array(interpolated_points_unrelaxed), 
            circumcenters_unrelaxed, 
            circumcenters_rbf)

###############################################################################
# Wasserstein Distance Calculation
###############################################################################

def wasserstein_distance(displacements1, density1, displacements2, density2, lattice_vector, index=0, structure_name=""):
    """Calculate the Wasserstein distance between two displacement distributions."""
    # Normalize densities
    density1_norm = density1 / np.sum(density1)
    density2_norm = density2 / np.sum(density2)
    
    # Create uniform distributions for comparison
    density1_uniform = np.ones_like(density1_norm) / len(density1_norm)
    density2_uniform = np.ones_like(density2_norm) / len(density2_norm)
    
    # Create periodic images for distance calculation
    offsets = np.array([i * lattice_vector[0] + j * lattice_vector[1] 
                       for i in range(-2, 3) for j in range(-2, 3)])
    
    def calculate_periodic_distance(points1, points2):
        """Calculate distances with periodic boundary conditions."""
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        diff_periodic = diff[np.newaxis, :, :, :] + offsets[:, np.newaxis, np.newaxis, :]
        return np.min(np.linalg.norm(diff_periodic, axis=-1), axis=0)
    
    # Calculate cost matrix and optimal transport
    cost_matrix = calculate_periodic_distance(displacements1, displacements2)
    transport_matrix = ot.emd(density1_norm, density2_norm, cost_matrix)
    transport_matrix_uniform = ot.emd(density1_uniform, density2_uniform, cost_matrix)
    
    # Calculate distances
    distance = np.sum(transport_matrix * cost_matrix)
    distance_uniform = np.sum(transport_matrix_uniform * cost_matrix)
    
    print(f"  • Weighted (Wasserstein) distance to {structure_name}: {distance:.6f} Å")
    print(f"  • Uniform distance to {structure_name}: {distance_uniform:.6f} Å")
    
    return distance, transport_matrix, distance_uniform, transport_matrix_uniform

###############################################################################
# Density Processing Functions
###############################################################################

def process_displacements(step, displacements, lattice_vector, bandwidth, total_structures):
    """Process displacement vectors to calculate density distributions."""
    print(f"Processing displacements {step + 1}/{total_structures}")
    
    # Extract 2D coordinates
    points_center = np.vstack([displacements[:, 0], displacements[:, 1]]).T
    points_all = pad_periodic_image(points_center, lattice_vector)
    
    # Calculate density using KDE
    kde = KernelDensity(bandwidth=bandwidth, metric='euclidean', kernel='gaussian')
    kde.fit(points_all.reshape(-1, 2))
    
    # Calculate densities
    density_center = np.exp(kde.score_samples(points_center))
    density_all = np.exp(kde.score_samples(points_all.reshape(-1, 2)))
    
    # Normalize densities
    normalization_factor = len(displacements) / np.sum(density_center)
    density_center *= normalization_factor
    density_all *= normalization_factor
    
    return density_center, density_all, points_center, points_all

def process_structure(args):
    """Process a single structure to calculate interpolated points."""
    step, structure, unrelaxed_positions_padded, lattice_vector, pristine_cell, total_structures = args
    print(f"\n[Structure {step+1}/{total_structures}] Starting structure processing...")
    
    # Get relaxed positions for type 0 atoms
    print(f"  → Extracting type 0 atom positions...")
    relaxed_positions = structure.positions[structure.arrays['atom_types'] == 0, :2]
    relaxed_positions_padded = pad_periodic_image(relaxed_positions, structure.cell[:2, :2]).reshape(-1, 2)
    print(f"    Found {len(relaxed_positions)} type 0 atoms")
    
    # Get query points for type 3 atoms
    print(f"  → Extracting type 3 atom positions...")
    query_points = structure.positions[structure.arrays['atom_types'] == 3, :2]
    print(f"    Found {len(query_points)} type 3 atoms")
    
    print("  → Starting Voronoi interpolation...")
    interpolated_points = voronoi_interpolation(
        relaxed_positions_padded,
        unrelaxed_positions_padded,
        query_points,
        pristine_cell
    )[0]  # Only take the first return value (pristine points)
    print("  ✓ Finished interpolation")
    
    return interpolated_points

def extract_distances(atom_displacements, lattice_vector, structure_names):
    """Calculate interlayer distances between structures using Wasserstein distance."""
    print("\n=== Starting Distance Calculations ===")
    
    # Calculate parameters for density estimation
    print("→ Initializing density estimation parameters...")
    pristine_cell = get_primitive_voronoi_cell(lattice_vector)
    bandwidth = np.linalg.norm(lattice_vector[0]) / np.sqrt(4 * len(atom_displacements[0]))
    cell_area = np.abs(np.linalg.det(lattice_vector))
    print(f"  • Bandwidth: {bandwidth:.4f} Å")
    print(f"  • Cell area: {cell_area:.4f} Å²")

    # Process all displacements to get density distributions
    print("\n→ Processing displacement distributions...")
    density_results = []
    for i, displacements in enumerate(atom_displacements):
        print(f"\n  Structure {i+1}/{len(atom_displacements)} ({structure_names[i]}):")
        result = process_displacements(i, displacements, lattice_vector, bandwidth, len(atom_displacements))
        density_results.append(result)

    # Unpack results
    print("\n→ Unpacking density results...")
    densities_center, densities_all, points_center, points_all = zip(*density_results)
    densities_center = np.array(densities_center)
    points_center = np.array(points_center)

    # Calculate distances
    print("\n→ Calculating Wasserstein distances...")
    distances_uniform = [0]  # First distance is to self (0)

    for i in range(1, len(atom_displacements)):
        print(f"\n  Computing distance between {structure_names[0]} and {structure_names[i]}:")
        if i == 1:
            initial_densities = densities_center[0]
            is_uniform = np.allclose(initial_densities, 1.0, rtol=1e-3, atol=1e-2)
            print(f"  • Initial density uniformity check: {'✓ Uniform' if is_uniform else '✗ Non-uniform'}")
            if not is_uniform:
                print(f"    Range: [{np.min(initial_densities):.6f}, {np.max(initial_densities):.6f}]")

        _, _, distance_uniform, _ = wasserstein_distance(
            points_center[0], densities_center[0],
            points_center[i], densities_center[i],
            lattice_vector, index=i, structure_name=structure_names[i]
        )
        distances_uniform.append(distance_uniform)

    print("\n=== Distance Calculations Complete ===")
    return distances_uniform

###############################################################################
# Main Execution
###############################################################################

if __name__ == "__main__":
    print("\n=== Starting Structure Analysis ===")
    
    # Load and prepare reference structure
    print("\n→ Loading reference structure...")
    reference_file = 'MoS2-WSe2_1p1deg_reference.xyz'
    reference_structure = read(reference_file, index=0, format='extxyz')
    lattice_vector = convert_lattice_string_to_array(reference_structure.info['base_lattice_0'])
    pristine_cell = get_primitive_voronoi_cell(lattice_vector)
    atom_types = reference_structure.arrays['atom_types']
    print(f"  ✓ Loaded reference structure with {len(atom_types)} atoms")
    print(f"  • Lattice parameters: a = {np.linalg.norm(lattice_vector[0]):.4f} Å, b = {np.linalg.norm(lattice_vector[1]):.4f} Å")

    # Load structures for comparison (UM1 = 6A cutoff, UM2 = 10A cutoff)
    print("\n→ Loading comparison structures...")
    structures = {
        'UM1': read("MoS2-WSe2_1p1deg_UM1_relaxed.xyz", index=-1, format='extxyz'),
        'UM2': read("MoS2-WSe2_1p1deg_UM2_relaxed.xyz", index=-1, format='extxyz'),
    }
    print("  ✓ Loaded all comparison structures")

    # Process each structure
    print("\n→ Processing structures...")
    for name, structure in structures.items():
        print(f"  Processing {name}...")
        structure.arrays['atom_types'] = atom_types
        structure.positions -= structure.positions[0]
        rotation_angle = -np.arctan(structure.cell[0, 1] / structure.cell[0, 0]) * 180 / np.pi
        structure.rotate(rotation_angle, 'z', rotate_cell=True)
        structure.wrap()
        print(f"  ✓ Completed {name}")

    # Prepare list of structures for processing
    structure_names = ['UM1', 'UM2']
    selected_structures = [structures[name] for name in structure_names]
    print(f"\n→ Selected {len(selected_structures)} structures for analysis:")
    for name in structure_names:
        print(f"  • {name}")

    # Process unrelaxed structure
    print("\n→ Processing unrelaxed structure...")
    unrelaxed_positions = structures['UM1'].positions[atom_types == 0, :2]
    unrelaxed_positions_padded = pad_periodic_image(unrelaxed_positions, structures['UM1'].cell[:2, :2]).reshape(-1, 2)
    print(f"  ✓ Processed unrelaxed structure with {len(unrelaxed_positions)} type 0 atoms")

    # Calculate displacements for each structure
    print("\n→ Calculating displacements...")
    displacements_list = []
    for i, structure in enumerate(selected_structures):
        args = (i, structure, unrelaxed_positions_padded, lattice_vector, pristine_cell, len(selected_structures))
        displacement = process_structure(args)
        displacements_list.append(displacement)
        print(f"  ✓ Completed {structure_names[i]}")

    print(f"\n✓ Successfully processed {len(displacements_list)} structures")
    
    # Calculate distances between structures
    print("\n→ Calculating final distances...")
    distances = extract_distances(displacements_list, lattice_vector, structure_names)
    print("\nFinal Results:")
    print("==============")
    for i, (name, distance) in enumerate(zip(structure_names, distances)):
        print(f"Distance between UM1 and {name}: {distance:.6f} Å")
    print("\n=== Analysis Complete ===\n")