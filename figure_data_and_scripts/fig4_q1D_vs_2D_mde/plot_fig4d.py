import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.style.use("../matplotlib.rc")

c_distances_list_1D_intralayer = np.load("MoS2-WSe2_1D_intralayer_disregistry_distances.npy")
c_distances_list_1D_interlayer = np.load("MoS2-WSe2_1D_interlayer_disregistry_distances.npy")
c_distances_list_2D_interlayer = np.load("MoS2-WSe2_2D_interlayer_disregistry_distances.npy")
c_distances_list_2D_intralayer = np.load("MoS2-WSe2_2D_intralayer_disregistry_distances.npy")


def group_by_first_column(data):
    """
    Group data by the first column and collect values from the third column.

    Parameters:
    data (list of lists): The input data, where each inner list is [i, j, value, ...]

    Returns:
    list of lists: Each inner list contains values from the third column for a unique first column value
    """
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


c_distances_list_1D_interlayer = group_by_first_column(c_distances_list_1D_interlayer)
c_distances_list_2D_interlayer = group_by_first_column(c_distances_list_2D_interlayer)
c_distances_list_1D_intralayer = group_by_first_column(c_distances_list_1D_intralayer)
c_distances_list_2D_intralayer = group_by_first_column(c_distances_list_2D_intralayer)

FIG_SIZE = (3.5, 2.5)

start_ind = 0
end_ind = 12
corruption_facs = np.logspace(-4.3, -0.3, 12)[start_ind:end_ind]


def calculate_iqr(data):
    q25 = np.percentile([x for x in data if x is not None], 25)
    q75 = np.percentile([x for x in data if x is not None], 75)
    return q25, q75


def calculate_error_bars(median, q25, q75):
    lower_error = abs(median - q25)
    upper_error = abs(q75 - median)
    return lower_error, upper_error


# --- Mean plot ---

c_distances_list_1D_interlayer_mean = [np.mean([x for x in c if x is not None]) for c in c_distances_list_1D_interlayer[start_ind:end_ind]]
c_distances_list_2D_interlayer_mean = [np.mean([x for x in c if x is not None]) for c in c_distances_list_2D_interlayer[start_ind:end_ind]]
c_distances_list_1D_intralayer_mean = [np.mean([x for x in c if x is not None]) for c in c_distances_list_1D_intralayer[start_ind:end_ind]]
c_distances_list_2D_intralayer_mean = [np.mean([x for x in c if x is not None]) for c in c_distances_list_2D_intralayer[start_ind:end_ind]]

c_distances_list_1D_interlayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_1D_interlayer[start_ind:end_ind]]
c_distances_list_1D_interlayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_1D_interlayer[start_ind:end_ind]]
c_distances_list_2D_interlayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_2D_interlayer[start_ind:end_ind]]
c_distances_list_2D_interlayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_2D_interlayer[start_ind:end_ind]]
c_distances_list_1D_intralayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_1D_intralayer[start_ind:end_ind]]
c_distances_list_1D_intralayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_1D_intralayer[start_ind:end_ind]]
c_distances_list_2D_intralayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_2D_intralayer[start_ind:end_ind]]
c_distances_list_2D_intralayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_2D_intralayer[start_ind:end_ind]]

intralayer_distance_1D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_1D_intralayer_mean, c_distances_list_1D_intralayer_q25, c_distances_list_1D_intralayer_q75)]
intralayer_distance_2D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_2D_intralayer_mean, c_distances_list_2D_intralayer_q25, c_distances_list_2D_intralayer_q75)]
interlayer_distance_1D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_1D_interlayer_mean, c_distances_list_1D_interlayer_q25, c_distances_list_1D_interlayer_q75)]
interlayer_distance_2D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_2D_interlayer_mean, c_distances_list_2D_interlayer_q25, c_distances_list_2D_interlayer_q75)]

plt.figure(figsize=FIG_SIZE, layout="constrained")

norm = colors.LogNorm(vmin=1e-5, vmax=1e0)

sc1 = plt.scatter(c_distances_list_1D_intralayer_mean,
                  c_distances_list_2D_intralayer_mean,
                  c=corruption_facs,
                  cmap=plt.colormaps['Reds'],
                  norm=norm,
                  label='Intralayer',
                  marker='o',
                  s=40,
                  edgecolors='black',
                  alpha=0.7, zorder=3)
plt.errorbar(c_distances_list_1D_intralayer_mean,
             c_distances_list_2D_intralayer_mean,
             xerr=np.array(intralayer_distance_1D_errors).T,
             yerr=np.array(intralayer_distance_2D_errors).T,
             fmt='none',
             ecolor='gray',
             elinewidth=1,
             capsize=3)

sc2 = plt.scatter(c_distances_list_1D_interlayer_mean,
                  c_distances_list_2D_interlayer_mean,
                  c=corruption_facs,
                  cmap=plt.colormaps['Reds'],
                  norm=norm,
                  label='Interlayer',
                  marker='s',
                  s=40,
                  edgecolors='black',
                  alpha=0.7, zorder=3)
plt.errorbar(c_distances_list_1D_interlayer_mean,
             c_distances_list_2D_interlayer_mean,
             xerr=np.array(interlayer_distance_1D_errors).T,
             yerr=np.array(interlayer_distance_2D_errors).T,
             fmt='none',
             ecolor='gray',
             elinewidth=1,
             capsize=3)

plt.plot([-0.05, 2], [-0.05, 2], 'grey', linestyle='-', linewidth=.5)

plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.tick_params(axis='both', which='both', direction='in')

major_ticks = np.logspace(-5, 1, 7)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks], fontsize=11)
ax.set_yticks(major_ticks)
ax.set_yticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks], fontsize=11)

plt.axis('scaled')
plt.xlim(3e-5, 2.)
plt.ylim(3e-5, 2.)

cbar = plt.colorbar(sc1, shrink=1.0, pad=0.01)
cbar.set_label('Model Weight Corruption Factor')
log_ticks = np.logspace(-5, 0, 6)
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])
for label in cbar.ax.get_yticklabels():
    label.set_fontsize(11)

plt.xlabel('1D Mean Disregistry Error (Å)')
plt.ylabel('2D Mean Disregistry Error (Å)')
plt.legend(loc='upper left', labelspacing=0.1, handletextpad=-0.2)
plt.grid(True, which='major', linestyle='-', linewidth=0.5)

plt.savefig('fig4d_mde_mean.pdf', dpi=300, transparent=True)
plt.close()


# --- Median plot ---

c_distances_list_1D_interlayer_median = [np.median([x for x in c if x is not None]) for c in c_distances_list_1D_interlayer[start_ind:end_ind]]
c_distances_list_1D_interlayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_1D_interlayer[start_ind:end_ind]]
c_distances_list_1D_interlayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_1D_interlayer[start_ind:end_ind]]

c_distances_list_2D_interlayer_median = [np.median([x for x in c if x is not None]) for c in c_distances_list_2D_interlayer[start_ind:end_ind]]
c_distances_list_2D_interlayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_2D_interlayer[start_ind:end_ind]]
c_distances_list_2D_interlayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_2D_interlayer[start_ind:end_ind]]

c_distances_list_1D_intralayer_median = [np.median([x for x in c if x is not None]) for c in c_distances_list_1D_intralayer[start_ind:end_ind]]
c_distances_list_1D_intralayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_1D_intralayer[start_ind:end_ind]]
c_distances_list_1D_intralayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_1D_intralayer[start_ind:end_ind]]

c_distances_list_2D_intralayer_median = [np.median([x for x in c if x is not None]) for c in c_distances_list_2D_intralayer[start_ind:end_ind]]
c_distances_list_2D_intralayer_q25 = [calculate_iqr(c)[0] for c in c_distances_list_2D_intralayer[start_ind:end_ind]]
c_distances_list_2D_intralayer_q75 = [calculate_iqr(c)[1] for c in c_distances_list_2D_intralayer[start_ind:end_ind]]

plt.figure(figsize=FIG_SIZE, layout="constrained")

intralayer_distance_1D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_1D_intralayer_median, c_distances_list_1D_intralayer_q25, c_distances_list_1D_intralayer_q75)]
intralayer_distance_2D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_2D_intralayer_median, c_distances_list_2D_intralayer_q25, c_distances_list_2D_intralayer_q75)]
interlayer_distance_1D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_1D_interlayer_median, c_distances_list_1D_interlayer_q25, c_distances_list_1D_interlayer_q75)]
interlayer_distance_2D_errors = [calculate_error_bars(median, q25, q75) for median, q25, q75 in zip(c_distances_list_2D_interlayer_median, c_distances_list_2D_interlayer_q25, c_distances_list_2D_interlayer_q75)]

norm = colors.LogNorm(vmin=1e-5, vmax=1e0)

sc1 = plt.scatter(c_distances_list_1D_intralayer_median,
                   c_distances_list_2D_intralayer_median,
                   c=corruption_facs,
                   cmap=plt.colormaps['Reds'],
                   norm=norm,
                   label='Intralayer',
                   marker='o',
                   s=40,
                   edgecolors='black',
                   alpha=.8, zorder=3)
plt.errorbar(c_distances_list_1D_intralayer_median,
             c_distances_list_2D_intralayer_median,
             xerr=np.array(intralayer_distance_1D_errors).T,
             yerr=np.array(intralayer_distance_2D_errors).T,
             fmt='none',
             ecolor='gray',
             elinewidth=1,
             capsize=3)

sc2 = plt.scatter(c_distances_list_1D_interlayer_median,
                   c_distances_list_2D_interlayer_median,
                   c=corruption_facs,
                   cmap=plt.colormaps['Reds'],
                   norm=norm,
                   label='Interlayer',
                   marker='s',
                   s=40,
                   edgecolors='black',
                   alpha=.8, zorder=3)
plt.errorbar(c_distances_list_1D_interlayer_median,
             c_distances_list_2D_interlayer_median,
             xerr=np.array(interlayer_distance_1D_errors).T,
             yerr=np.array(interlayer_distance_2D_errors).T,
             fmt='none',
             ecolor='gray',
             elinewidth=1,
             capsize=3)

plt.plot([5e-5, 2.], [5e-5, 2.], 'grey', linestyle='-', linewidth=.5, zorder=0)

plt.xscale('log')
plt.yscale('log')

ax = plt.gca()
ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
ax.tick_params(axis='both', which='both', direction='in')

major_ticks = np.logspace(-5, 1, 7)
ax.set_xticks(major_ticks)
ax.set_xticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks], fontsize=11)
ax.set_yticks(major_ticks)
ax.set_yticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in major_ticks], fontsize=11)

plt.axis('scaled')
plt.xlim(3e-5, 2.)
plt.ylim(3e-5, 2.)

cbar = plt.colorbar(sc1, pad=0.00, shrink=1.0)
cbar.set_label('Model Weight Corruption Factor')
log_ticks = np.logspace(-5, 0, 6)
cbar.set_ticks(log_ticks)
cbar.ax.tick_params(which='both', direction='in')
cbar.set_ticklabels([f'$10^{{{int(np.log10(x))}}}$' for x in log_ticks])
for label in cbar.ax.get_yticklabels():
    label.set_fontsize(11)

plt.xlabel('1D Median Disregistry Error (Å)')
plt.ylabel('2D Median Disregistry Error (Å)')
plt.legend(loc='upper left', labelspacing=0.1, handletextpad=-0.2)

plt.savefig('fig4d_mde_median.pdf', dpi=300, transparent=True)
plt.close()
