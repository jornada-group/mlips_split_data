"""
Figure 3e: Voronoi-centered disregistry vector distribution for UM1 (6 Ang cutoff).

Reads the UM1-relaxed 1.1-deg twisted MoS2/WSe2 structure and the
configuration-space energy sweep, computes Voronoi-interpolated disregistry
vectors for all W sites, and plots them mapped onto the pristine MoS2
Wigner-Seitz cell.  A cached .npy file is written on first run for speed.
"""
import time
import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import cdist
import os
import matplotlib.colors as mcolors

plt.style.use("../matplotlib.rc")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

displacement_magnitude_cmap = plt.cm.YlGn


def convert_lattice_string_to_array(string):
    return np.array([float(x) for x in string.strip('[]').split()]).reshape(2, 2)

def pad_periodic_image(pos, box, n_a1=1, n_a2=1):
    i_range = np.concatenate((np.arange(0, n_a1 + 1), np.arange(-n_a1, 0)))
    j_range = np.concatenate((np.arange(0, n_a2 + 1), np.arange(-n_a2, 0)))
    i, j = np.meshgrid(i_range, j_range)
    offsets = i.flatten()[:, np.newaxis] * box[0] + j.flatten()[:, np.newaxis] * box[1]
    padded_pos = pos[np.newaxis, :, :] + offsets[:, np.newaxis, :]
    return np.vstack((pos, padded_pos[1:].reshape(-1, 2)))

def get_primitive_voronoi_cell(A1):
    x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    points = np.column_stack((x.ravel(), y.ravel()))
    lattice_points = points @ A1
    vor = Voronoi(lattice_points)
    central_region = vor.regions[vor.point_region[4]]
    return vor.vertices[central_region] if -1 not in central_region else None

def voronoi_interpolation(relaxed_points, unrelaxed_points,
                          query_points, pristine_voronoi_vertices):
    def find_voronoi_cell(query_point, points, tri, circumcenters):
        distances = np.sum((points - query_point)**2, axis=1)
        nearest_point_index = np.argmin(distances)
        simplices_containing_point = np.where(
            (tri.simplices == nearest_point_index).any(axis=1))[0]
        cell_indices = simplices_containing_point[np.argsort(np.arctan2(
            circumcenters[simplices_containing_point][:, 1] - points[nearest_point_index, 1],
            circumcenters[simplices_containing_point][:, 0] - points[nearest_point_index, 0]
        ))]
        return cell_indices, nearest_point_index

    def calculate_circumcenter(triangle):
        a, b, c = triangle
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-10:
            return np.mean(triangle, axis=0)
        ux = ((a[0]**2 + a[1]**2) * (b[1] - c[1]) + (b[0]**2 + b[1]**2) * (c[1] - a[1]) +
              (c[0]**2 + c[1]**2) * (a[1] - b[1])) / d
        uy = ((a[0]**2 + a[1]**2) * (c[0] - b[0]) + (b[0]**2 + b[1]**2) * (a[0] - c[0]) +
              (c[0]**2 + c[1]**2) * (b[0] - a[0])) / d
        return np.array([ux, uy])

    def map_to_pristine_voronoi(points, original_vertices, pristine_vertices,
                                original_center, pristine_center):
        original_vertices_centered = original_vertices - (original_center - pristine_center)
        pristine_vertices_centered = pristine_vertices - pristine_center
        cost_matrix = cdist(original_vertices_centered, pristine_vertices_centered)
        _, col_ind = linear_sum_assignment(cost_matrix)
        pristine_vertices_matched = pristine_vertices_centered[col_ind]
        tps = RBFInterpolator(original_vertices_centered, pristine_vertices_matched,
                              kernel='thin_plate_spline', smoothing=0)
        points_centered = points - (original_center - pristine_center)
        transformed_points = tps(points_centered)
        return transformed_points + pristine_center

    tri_relaxed = Delaunay(relaxed_points)
    circumcenters_relaxed = np.array([calculate_circumcenter(relaxed_points[s])
                                      for s in tri_relaxed.simplices])

    tri_unrelaxed = Delaunay(unrelaxed_points)
    circumcenters_unrelaxed = np.array([calculate_circumcenter(unrelaxed_points[s])
                                        for s in tri_unrelaxed.simplices])

    rbf_interpolator = RBFInterpolator(unrelaxed_points, relaxed_points,
                                       kernel='thin_plate_spline')
    circumcenters_rbf = rbf_interpolator(circumcenters_unrelaxed)

    pristine_center = np.mean(pristine_voronoi_vertices, axis=0)
    pristine_voronoi_vertices_cell = np.vstack([pristine_voronoi_vertices, pristine_center])
    interpolated_points_pristine = []
    interpolated_points_unrelaxed = []

    for query_point in query_points:
        voronoi_indices, nearest_point_index = find_voronoi_cell(
            query_point, relaxed_points, tri_unrelaxed, circumcenters_rbf)
        voronoi_vertices_relaxed = np.vstack(
            [relaxed_points[nearest_point_index], circumcenters_rbf[voronoi_indices]])
        voronoi_vertices_unrelaxed = np.vstack(
            [unrelaxed_points[nearest_point_index], circumcenters_unrelaxed[voronoi_indices]])

        rbf_voronoi = RBFInterpolator(voronoi_vertices_relaxed, voronoi_vertices_unrelaxed)
        interpolated_point = rbf_voronoi(query_point.reshape(1, -1))[0]

        mapped_point_pristine = map_to_pristine_voronoi(
            interpolated_point.reshape(1, -1),
            voronoi_vertices_unrelaxed,
            pristine_voronoi_vertices_cell,
            unrelaxed_points[nearest_point_index],
            pristine_center
        )[0]

        interpolated_points_pristine.append(mapped_point_pristine)
        interpolated_points_unrelaxed.append(interpolated_point)

    return (np.array(interpolated_points_pristine),
            np.array(interpolated_points_unrelaxed),
            circumcenters_unrelaxed,
            circumcenters_rbf)


def create_displacement_plot(displacements_list, A1, confined_displacements_energy, energies,
                             output_file='fig3e_disregistry_UM1.pdf',
                             use_magnitude_colors=True):
    pristine_cell = get_primitive_voronoi_cell(A1)
    centroid = np.mean(pristine_cell, axis=0)
    energies = energies * 1000

    x = np.linspace(centroid[0] - 1.5*np.linalg.norm(A1[0]),
                    centroid[0] + 1.5*np.linalg.norm(A1[0]), 200)
    y = np.linspace(centroid[1] - 1.5*np.linalg.norm(A1[1]),
                    centroid[1] + 1.5*np.linalg.norm(A1[1]), 200)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.ravel(), Y.ravel()))

    n_repeats = 1
    repeated_displacements, repeated_energies = [], []
    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            repeated_displacements.append(confined_displacements_energy + offset)
            repeated_energies.append(energies)

    repeated_displacements = np.vstack(repeated_displacements)
    repeated_energies = np.concatenate(repeated_energies)

    rbf = RBFInterpolator(repeated_displacements + centroid, repeated_energies,
                          kernel='thin_plate_spline', smoothing=0.1)
    interpolated_energies = rbf(grid_points).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(1.25, 1.25), dpi=300, layout='constrained')
    extent = [x.min(), x.max(), y.min(), y.max()]
    energy_norm = mcolors.Normalize(vmin=np.min(energies), vmax=np.max(energies))
    ax.imshow(interpolated_energies, extent=extent, origin='lower', cmap='gist_gray',
              norm=energy_norm, aspect='equal', alpha=0.7)

    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            cell = pristine_cell + offset
            ax.plot(np.append(cell[:, 0], cell[0, 0]),
                    np.append(cell[:, 1], cell[0, 1]),
                    'lightskyblue', linewidth=1, alpha=0.4, zorder=6)

    for i in range(-n_repeats, n_repeats+1):
        for j in range(-n_repeats, n_repeats+1):
            offset = i * A1[0] + j * A1[1]
            for idx, displacement in enumerate(displacements_list):
                end_point = centroid + displacement + offset
                if use_magnitude_colors:
                    magnitude = np.linalg.norm(displacement)
                    max_magnitude = np.max([np.linalg.norm(d) for d in displacements_list])
                    color = displacement_magnitude_cmap(magnitude / max_magnitude)
                    ax.scatter(end_point[0], end_point[1],
                              color=color, s=2, alpha=1.0, marker='h', zorder=5)
                else:
                    atom_norm = plt.Normalize(0, len(displacements_list)-1)
                    ax.scatter(end_point[0], end_point[1],
                              c=[[idx]], s=2, alpha=1.0, marker='h', zorder=5)

    padding_factor = .6
    ax.set_xlim(centroid[0] - padding_factor*np.linalg.norm(A1[0]),
                centroid[0] + padding_factor*np.linalg.norm(A1[0]))
    ax.set_ylim(centroid[1] - padding_factor*np.linalg.norm(A1[0]),
                centroid[1] + padding_factor*np.linalg.norm(A1[1]))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    plt.savefig(output_file, dpi=300, transparent=True)
    plt.close(fig)
    print(f"Figure saved as '{output_file}'")


def main():
    start_time = time.time()
    reference_structure = read("MoS2-WSe2_1p1deg_reference.xyz", index='-1', format='extxyz')
    A1 = convert_lattice_string_to_array(reference_structure.info['base_lattice_0'])
    atom_types = reference_structure.arrays['atom_types']
    pristine_cell = get_primitive_voronoi_cell(A1)

    structure = read('MoS2-WSe2_1p1deg_UM1_relaxed.xyz', index='-1', format='extxyz')
    structure.arrays['atom_types'] = atom_types

    energy_structures = read('MoS2-WSe2_config_space_energies.xyz', index=':', format='extxyz')

    confined_displacements_energy = []
    energies = []
    for es in energy_structures:
        padded_pos = pad_periodic_image(
            es.positions[es.arrays['atom_types'] == 0, :2],
            es.cell[:2, :2], n_a1=3, n_a2=3)
        query_pts = es.positions[es.arrays['atom_types'] == 3, :2]
        pts_pristine, _, _, _ = voronoi_interpolation(
            padded_pos, padded_pos, query_pts, pristine_cell)
        confined_displacements_energy.append(pts_pristine)
        energies.append(es.get_potential_energy())

    confined_displacements_energy = np.array(confined_displacements_energy).reshape(-1, 2)
    energies = np.array(energies)
    energies -= np.max(energies)

    cache_file = 'MoS2-WSe2_1p1deg_displacements_UM1.npy'
    if os.path.isfile(cache_file):
        displacements_list = np.load(cache_file)
        print(f"Loaded cached displacements from '{cache_file}'")
    else:
        padded_pos = pad_periodic_image(
            structure.positions[atom_types == 0, :2],
            structure.cell[:2, :2])
        query_pts = structure.positions[atom_types == 3, :2]
        pts_pristine, _, _, _ = voronoi_interpolation(
            padded_pos, padded_pos, query_pts, pristine_cell)
        displacements_list = pts_pristine
        np.save(cache_file, displacements_list)
        print(f"Saved displacements cache to '{cache_file}'")

    create_displacement_plot(displacements_list, A1,
                             confined_displacements_energy, energies)
    print(f"Done in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
