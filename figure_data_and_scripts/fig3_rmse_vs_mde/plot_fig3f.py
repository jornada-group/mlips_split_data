"""
Figure 3f: MDE (mean disregistry error) vs RMSE for corrupted MLIPs.

Two scatter plots are produced:
  fig3f_energy_rmse_vs_mde.pdf  — x-axis: energy RMSE (meV/atom)
  fig3f_force_rmse_vs_mde.pdf   — x-axis: force RMSE (meV/Ang)

Both show intralayer (circles) and interlayer (squares) corrupted MLIP
predictions as a function of model-weight corruption factor (colorbar).

Data layout:
  interlayer_corrupted/energy_baseline_all_test_2_inter_corrupted_SEED_<j>_facidx_<i>.xyz
  intralayer_corrupted/energy_baseline_all_test_2_intra_corrupted_SEED_<j>_facidx_<i>.xyz
  interlayer_reference/energy_baseline_all_test_2_inter_reference.xyz
  intralayer_reference/energy_baseline_all_test_2_intra_reference.xyz
  MoS2-WSe2_interlayer_disregistry_distances.npy
  MoS2-WSe2_intralayer_disregistry_distances.npy
"""
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import os
import matplotlib.colors as colors
import matplotlib.colors as mcolors

FIG_WIDTH = 4.2
FIG_HEIGHT = 3.5

plt.style.use("../matplotlib.rc")


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_energies(file_path_name):
    energies_list = []
    for i in range(12):
        energies = []
        for j in range(10):
            file_path = f"{file_path_name}_corrupted_SEED_{j}_facidx_{i}.xyz"
            if os.path.exists(file_path):
                relaxed_structures = read(file_path, index=":")
                energies.append([a.get_potential_energy() for a in relaxed_structures])
            else:
                print(f"File {file_path} does not exist.")
                energies.append(None)
        energies_list.append(energies)
    np.save(f'{file_path_name}_energies_list.npy', energies_list)
    return energies_list

def extract_forces(file_path_name):
    forces_list = []
    for i in range(12):
        forces = []
        for j in range(10):
            file_path = f"{file_path_name}_corrupted_SEED_{j}_facidx_{i}.xyz"
            if os.path.exists(file_path):
                relaxed_structures = read(file_path, index=":")
                forces.append([a.get_forces() for a in relaxed_structures])
            else:
                print(f"File {file_path} does not exist.")
                forces.append(None)
        forces_list.append(forces)
    np.save(f'{file_path_name}_forces_list.npy', forces_list)
    return forces_list

def extract_forces_norm(file_path_name):
    forces_list = []
    for i in range(12):
        forces = []
        for j in range(10):
            file_path = f"{file_path_name}_corrupted_SEED_{j}_facidx_{i}.xyz"
            if os.path.exists(file_path):
                relaxed_structures = read(file_path, index=":")
                forces.append([np.linalg.norm(a.get_forces(), axis=-1) for a in relaxed_structures])
            else:
                print(f"File {file_path} does not exist.")
                forces.append(None)
        forces_list.append(forces)
    np.save(f'{file_path_name}_forces_list_norm.npy', forces_list)
    return forces_list


# ---------------------------------------------------------------------------
# Load / cache corrupted trajectories
# ---------------------------------------------------------------------------

def load_or_extract(file_path_name, suffix, extractor_fn):
    cache = f'{file_path_name}_{suffix}.npy'
    if os.path.exists(cache):
        return np.load(cache, allow_pickle=True)
    return extractor_fn(file_path_name)


inter_base = "interlayer_corrupted/energy_baseline_all_test_2_inter"
intra_base = "intralayer_corrupted/energy_baseline_all_test_2_intra"

energies_interlayer_list = load_or_extract(inter_base, "energies_list", extract_energies)
forces_interlayer_list   = load_or_extract(inter_base, "forces_list",   extract_forces)
forces_interlayer_norm   = load_or_extract(inter_base, "forces_list_norm", extract_forces_norm)

energies_intralayer_list = load_or_extract(intra_base, "energies_list", extract_energies)
forces_intralayer_list   = load_or_extract(intra_base, "forces_list",   extract_forces)
forces_intralayer_norm   = load_or_extract(intra_base, "forces_list_norm", extract_forces_norm)

# Reference DFT energies and forces
ref_inter_structs = read("interlayer_reference/energy_baseline_all_test_2_inter_reference.xyz", index=":")
ref_energies_interlayer = [a.get_potential_energy() for a in ref_inter_structs]
ref_forces_interlayer   = [a.get_forces() for a in ref_inter_structs]
ref_forces_interlayer_norm = [np.linalg.norm(a.get_forces(), axis=-1) for a in ref_inter_structs]

ref_intra_structs = read("intralayer_reference/energy_baseline_all_test_2_intra_reference.xyz", index=":")
ref_energies_intralayer = [a.get_potential_energy() for a in ref_intra_structs]
ref_forces_intralayer   = [a.get_forces() for a in ref_intra_structs]
ref_forces_intralayer_norm = [np.linalg.norm(a.get_forces(), axis=-1) for a in ref_intra_structs]

# Pre-computed Wasserstein MDE data
c_distances_list_2D_interlayer_raw = np.load("MoS2-WSe2_interlayer_disregistry_distances.npy")
c_distances_list_2D_intralayer_raw = np.load("MoS2-WSe2_intralayer_disregistry_distances.npy")


# ---------------------------------------------------------------------------
# Grouping helper
# ---------------------------------------------------------------------------

def group_by_first_column(data):
    i_counter = 0
    data_list = []
    seed_list = []
    for row in data:
        current_i = row[0]
        value = row[2]
        if current_i != i_counter:
            if seed_list:
                data_list.append(seed_list)
            seed_list = []
            i_counter = current_i
        seed_list.append(value)
    if seed_list:
        data_list.append(seed_list)
    return data_list


c_distances_list_2D_interlayer = group_by_first_column(c_distances_list_2D_interlayer_raw)
c_distances_list_2D_intralayer = group_by_first_column(c_distances_list_2D_intralayer_raw)


# ---------------------------------------------------------------------------
# RMSE helpers
# ---------------------------------------------------------------------------

def get_ediffs(energies_list, ref_energies_list, n_atoms):
    e_diffs = []
    for i in range(12):
        e_diff = []
        for j in range(10):
            if energies_list[i][j] is not None:
                diff = np.array(energies_list[i][j]) - np.array(ref_energies_list)
                diff_mean = np.sqrt(np.mean(np.square(diff), axis=-1))
                e_diff.append(diff_mean)
        e_diffs.append(e_diff)
    return e_diffs


def get_fdiffs(forces_list, ref_forces_list):
    f_diffs = []
    for i in range(12):
        f_diff = []
        for j in range(10):
            if forces_list[i][j] is not None:
                pred = np.array(forces_list[i][j])
                ref  = np.array(ref_forces_list)
                diff = pred - ref
                diff_norm = np.linalg.norm(diff, axis=(1, 2))
                diff_rmse = np.sqrt(np.mean(diff_norm**2, axis=-1))
                f_diff.append(diff_rmse)
        f_diffs.append(f_diff)
    return f_diffs


# interlayer: 12 atoms per structure; intralayer: 27 atoms per structure
e_diffs_interlayer = get_ediffs(energies_interlayer_list, ref_energies_interlayer, n_atoms=12)
e_diffs_intralayer = get_ediffs(energies_intralayer_list, ref_energies_intralayer, n_atoms=27)

e_diffs_interlayer_seed = [np.array(ed) * 1000 / 12 for ed in e_diffs_interlayer]
e_diffs_intralayer_seed = [np.array(ed) * 1000 / 27 for ed in e_diffs_intralayer]

f_diffs_interlayer = get_fdiffs(forces_interlayer_list, ref_forces_interlayer)
f_diffs_intralayer = get_fdiffs(forces_intralayer_list, ref_forces_intralayer)

f_diffs_interlayer_seed = [np.array(fd) * 1000 for fd in f_diffs_interlayer]
f_diffs_intralayer_seed = [np.array(fd) * 1000 for fd in f_diffs_intralayer]


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

start_ind, end_ind = 0, 12
corruption_facs = np.logspace(-4.3, -0.3, 12)[start_ind:end_ind]


def calculate_iqr(data):
    valid = [x for x in data if x is not None]
    return np.percentile(valid, 25), np.percentile(valid, 75)


def stats(data):
    median = [np.median([x for x in c if x is not None]) for c in data[start_ind:end_ind]]
    q25    = [calculate_iqr(c)[0] for c in data[start_ind:end_ind]]
    q75    = [calculate_iqr(c)[1] for c in data[start_ind:end_ind]]
    return median, q25, q75


def calculate_error_bars(median, q25, q75):
    return abs(median - q25), abs(q75 - median)


e_inter_median, e_inter_q25, e_inter_q75 = stats(e_diffs_interlayer_seed)
e_intra_median, e_intra_q25, e_intra_q75 = stats(e_diffs_intralayer_seed)

d_inter_median, d_inter_q25, d_inter_q75 = stats(c_distances_list_2D_interlayer)
d_intra_median, d_intra_q25, d_intra_q75 = stats(c_distances_list_2D_intralayer)

f_inter_median, f_inter_q25, f_inter_q75 = stats(f_diffs_interlayer_seed)
f_intra_median, f_intra_q25, f_intra_q75 = stats(f_diffs_intralayer_seed)

e_inter_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(e_inter_median, e_inter_q25, e_inter_q75)]
e_intra_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(e_intra_median, e_intra_q25, e_intra_q75)]
d_inter_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(d_inter_median, d_inter_q25, d_inter_q75)]
d_intra_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(d_intra_median, d_intra_q25, d_intra_q75)]
f_inter_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(f_inter_median, f_inter_q25, f_inter_q75)]
f_intra_err = [calculate_error_bars(m, q25, q75)
               for m, q25, q75 in zip(f_intra_median, f_intra_q25, f_intra_q75)]


# ---------------------------------------------------------------------------
# Gradient shading helper
# ---------------------------------------------------------------------------

def generate_gradient(color_hex, width=256, height=100, horizontal=False):
    rgba = mcolors.to_rgba(color_hex)
    gradient = np.zeros((height, width, 4))
    gradient[:, :, :3] = rgba[:3]
    if horizontal:
        gradient[:, :, 3] = np.linspace(0.0, 0.5, height)[:, np.newaxis]
    else:
        gradient[:, :, 3] = np.linspace(0.0, 0.5, width)[np.newaxis, :]
    return gradient


# ---------------------------------------------------------------------------
# Figure 1: Energy RMSE vs MDE
# ---------------------------------------------------------------------------

norm = colors.LogNorm(vmin=1e-5, vmax=1e0)

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")

sc1 = plt.scatter(e_intra_median, d_intra_median,
                  c=corruption_facs, cmap=plt.colormaps['Reds'], norm=norm,
                  label='Intralayer', marker='o', s=40,
                  edgecolors='black', alpha=.8, zorder=3)
plt.errorbar(e_intra_median, d_intra_median,
             xerr=np.array(e_intra_err).T, yerr=np.array(d_intra_err).T,
             fmt='none', ecolor='gray', elinewidth=1, capsize=3)

sc2 = plt.scatter(e_inter_median, d_inter_median,
                  c=corruption_facs, cmap=plt.colormaps['Reds'], norm=norm,
                  label='Interlayer', marker='s', s=40,
                  edgecolors='black', alpha=.8, zorder=3)
plt.errorbar(e_inter_median, d_inter_median,
             xerr=np.array(e_inter_err).T, yerr=np.array(d_inter_err).T,
             fmt='none', ecolor='gray', elinewidth=1, capsize=3)

plt.xlim(1e-3, 1.5e4)
x_min, x_max = plt.xlim()

# Vertical threshold bands: interlayer ≤ 4 meV/atom (green), intralayer ≤ 200 meV/atom (orange)
for inter, desired_value, col in [(True, 5, "#66c2a5"), (False, 2e2, "#fc8d62")]:
    gradient = generate_gradient(col)
    plt.imshow(gradient, aspect="auto",
               extent=[desired_value * 1e-3, desired_value, 0.0, 1], origin='lower')
    plt.vlines(desired_value, 0.0, 1, color=col, linestyle="--", linewidth=1.5, zorder=0)

# Horizontal MDE threshold: ≤ 0.01 Å
grey_col = "#808080"
h_gradient = generate_gradient(grey_col, horizontal=True)
plt.imshow(h_gradient, aspect="auto",
           extent=[x_min, x_max, 1e-2 * 0.1, 1e-2], origin='lower')
plt.axhline(y=1e-2, color=grey_col, linestyle="--", linewidth=1.5, zorder=0)

plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.tick_params(axis='both', which='both', direction='in')
major_ticks = np.logspace(-3, 4, 8)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks])

cbar = plt.colorbar(sc1, pad=-0.01)
cbar.set_label('Model Weight Corruption Factor')
log_ticks = np.logspace(-5, 0, 6)
cbar.set_ticks(log_ticks)
cbar.ax.tick_params(direction='in', which='both')
cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])

plt.xlabel('Corrupted MLIP Energy RMSE (meV/atom)')
plt.ylabel('2D Struct. Mean Disregistry Error (Å)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.33), ncol=2)

plt.savefig('fig3f_energy_rmse_vs_mde.pdf', dpi=300, transparent=True)
plt.close()
print("Saved fig3f_energy_rmse_vs_mde.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Force RMSE vs MDE
# ---------------------------------------------------------------------------

plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), layout="constrained")

sc1 = plt.scatter(f_intra_median, d_intra_median,
                  c=corruption_facs, cmap=plt.colormaps['Reds'], norm=norm,
                  label='Intralayer', marker='o', s=40,
                  edgecolors='black', alpha=.8, zorder=3)
plt.errorbar(f_intra_median, d_intra_median,
             xerr=np.array(f_intra_err).T, yerr=np.array(d_intra_err).T,
             fmt='none', ecolor='gray', elinewidth=1, capsize=3)

sc2 = plt.scatter(f_inter_median, d_inter_median,
                  c=corruption_facs, cmap=plt.colormaps['Reds'], norm=norm,
                  label='Interlayer', marker='s', s=40,
                  edgecolors='black', alpha=.8, zorder=3)
plt.errorbar(f_inter_median, d_inter_median,
             xerr=np.array(f_inter_err).T, yerr=np.array(d_inter_err).T,
             fmt='none', ecolor='gray', elinewidth=1, capsize=3)

plt.xlim(1e-2, 1e5)
x_min, x_max = plt.xlim()

# Vertical threshold bands: interlayer ≤ 200 meV/Ang (green), intralayer ≤ 4000 meV/Ang (orange)
for inter, desired_value, col in [(True, 2e2, "#66c2a5"), (False, 4e3, "#fc8d62")]:
    gradient = generate_gradient(col)
    plt.imshow(gradient, aspect="auto",
               extent=[desired_value * 1e-3, desired_value, 0.0, 1], origin='lower')
    plt.vlines(desired_value, 0.0, 1, color=col, linestyle="--", linewidth=1.5, zorder=0)

# Horizontal MDE threshold
grey_col = "#808080"
h_gradient = generate_gradient(grey_col, horizontal=True)
plt.imshow(h_gradient, aspect="auto",
           extent=[x_min, x_max, 1e-2 * 0.1, 1e-2], origin='lower')
plt.axhline(y=1e-2, color=grey_col, linestyle="--", linewidth=1.5, zorder=0)

plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.tick_params(axis='both', which='both', direction='in')
major_ticks = np.logspace(-2, 5, 8)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks])

cbar = plt.colorbar(sc1, pad=-0.04)
cbar.set_label('Model Weight Corruption Factor')
log_ticks = np.logspace(-5, 0, 6)
cbar.set_ticks(log_ticks)
cbar.ax.tick_params(direction='in', which='both')
cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])

plt.xlabel('Corrupted MLIP Force RMSE (meV/Å)')
plt.ylabel('2D Struct. Mean Disregistry Error (Å)')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.33), ncol=2)

plt.savefig('fig3f_force_rmse_vs_mde.pdf', dpi=300, transparent=True)
plt.close()
print("Saved fig3f_force_rmse_vs_mde.pdf")
