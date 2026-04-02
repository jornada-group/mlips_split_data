import numpy as np
from pathlib import Path
import os
from ase.io import read
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patheffects import Stroke, Normal
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib as mpl

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# Explicit model file order matches the x-axis left-to-right sequence.
# Intra: SW | Unified (rcut=6) | Split Allegro (rcut=6) | Unified (rcut=10) | Split MACE
INTRA_MODEL_FILES = [
    "sw_lammps_wse2_validation.xyz",
    "baseline_02_all_valid.xyz",
    "split_intra_03_small_all_valid.xyz",
    "baseline_01_all_valid.xyz",
    "split_mace_intra_wse2.xyz",
]
# Inter: KC | Unified | Split small NN | Split large NN | Split MACE inter
INTER_MODEL_FILES = [
    "kc_lammps_mos2_wse2_validation.xyz",
    "baseline_01_all_valid.xyz",
    "split_inter_02_small_all_valid.xyz",
    "split_inter_03_all_valid.xyz",
    "split_inter_01_all_valid.xyz",
]

FIGHEIGHT = 8
FIGWIDTH = 4.2

colorsall = [
    "#e5c494",
    "#a6d854",
    "#8da0cb",
    "#e78ac3",
    "#DA8BC3",
    "#64B5CD",
]
colorintra = [
    colorsall[0],
    colorsall[1],
    colorsall[2],
    colorsall[3],
    colorsall[2],
]
colorinter = [
    colorsall[0],
    colorsall[1],
    colorsall[2],
    colorsall[3],
    colorsall[3],
]


def generate_gradient(color_hex, width, height):
    rgba_color = mcolors.to_rgba(color_hex)
    gradient = np.zeros((height, width, 4))
    gradient[:, :, 0] = rgba_color[0]
    gradient[:, :, 1] = rgba_color[1]
    gradient[:, :, 2] = rgba_color[2]
    gradient[:, :, 3] = np.linspace(0.5, 0, height)[:, np.newaxis]
    return gradient


def get_property(name: str, atom):
    if name == "energy":
        if "MACE_energy" not in list(atom.info.keys()):
            return atom.get_total_energy()
        else:
            return atom.info["MACE_energy"]
    elif name == "energy_per_atom":
        if "MACE_energy" not in list(atom.info.keys()):
            return atom.get_total_energy() / len(atom) * 1e3
        else:
            return atom.info["MACE_energy"] / len(atom) * 1e3
    elif name == "compwisenorm_forces":
        if "MACE_forces" not in list(atom.arrays.keys()):
            return np.linalg.norm(atom.get_forces()) * 1e3
        else:
            return np.linalg.norm(atom.arrays["MACE_forces"]) * 1e3
    elif name == "max_force_comp":
        if "MACE_forces" not in list(atom.arrays.keys()):
            return np.max(atom.get_forces())
        else:
            return np.max(atom.arrays["MACE_forces"])
    elif name == "all_forces":
        if "MACE_forces" not in list(atom.arrays.keys()):
            return atom.get_forces() * 1e3
        else:
            return atom.arrays["MACE_forces"] * 1e3


def get_property_legend(name: str):
    if name == "energy":
        return "Residual Energy (eV)"
    elif name == "energy_per_atom":
        return "Residual Energy per Atom (meV/atom)"
    elif name == "all_forces":
        return "Residual Norm of all Forces (meV/Å)"
    elif name == "max_force_comp":
        return "Residual of Max of Forces (meV/Å)"


def get_residuals(property_name: str, energy_prop: bool):
    ground_truth_dset = DATA_DIR / "baseline_all_test_2.xyz"
    ground_truth_strucs = read(str(ground_truth_dset), ":", format="extxyz")
    ground_truth_es = np.array(
        [
            get_property(property_name, atom=at)
            for at in ground_truth_strucs
            if len(at) == 27 and at.numbers[0] == 74
        ]
    )

    if energy_prop:
        min_e_struc = np.argmin(ground_truth_es)
        ground_truth_es = ground_truth_es - ground_truth_es[min_e_struc]

    if len(ground_truth_es.shape) > 1:
        multdimprop = True

    residualall = []
    rmseall = []
    model_es = np.mean(ground_truth_es) * np.ones(ground_truth_es.shape)

    if len(ground_truth_es.shape) > 1:
        ndata = ground_truth_es.shape[0]
        residuals = np.linalg.norm(
            np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
        )
        rmse = np.median(
            np.linalg.norm(
                np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
            )
        )
    else:
        residuals = np.abs(ground_truth_es - model_es)
        rmse = np.median(np.abs(ground_truth_es - model_es))

    residualall.append(residuals)
    rmseall.append(rmse)

    for fname in INTRA_MODEL_FILES:
        model_strucs = read(str(DATA_DIR / "intra_WSe2" / fname), ":", format="extxyz")
        model_es = np.array(
            [
                get_property(property_name, atom=at)
                for at in model_strucs
                if len(at) == 27 and (at.numbers[0] == 3 or at.numbers[0] == 74)
            ]
        )

        if energy_prop:
            model_es = model_es - model_es[min_e_struc]

        if len(ground_truth_es.shape) > 1:
            ndata = ground_truth_es.shape[0]
            residuals = np.linalg.norm(
                np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
            )
            rmse = np.median(
                np.linalg.norm(
                    np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
                )
            )
        else:
            residuals = np.abs(ground_truth_es - model_es)
            rmse = np.median(np.abs(ground_truth_es - model_es))

        residualall.append(residuals)
        rmseall.append(rmse)

    data = np.array(residualall).T
    rmse_data = np.array(rmseall)
    return data, rmse_data


def get_residuals_inter(property_name: str, energy_prop: bool):
    ground_truth_dset = DATA_DIR / "baseline_all_test_2.xyz"
    ground_truth_strucs = read(str(ground_truth_dset), ":", format="extxyz")
    ground_truth_es = np.array(
        [
            get_property(property_name, atom=at)
            for at in ground_truth_strucs
            if len(at) == 12
        ]
    )

    if energy_prop:
        min_e_struc = np.argmin(ground_truth_es)
        ground_truth_es = ground_truth_es - ground_truth_es[min_e_struc]

    residualall = []
    rmseall = []
    model_es = np.mean(ground_truth_es) * np.ones(ground_truth_es.shape)

    if len(ground_truth_es.shape) > 1:
        ndata = ground_truth_es.shape[0]
        residuals = np.linalg.norm(
            np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
        )
        rmse = np.median(
            np.linalg.norm(
                np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
            )
        )
    else:
        residuals = np.abs(ground_truth_es - model_es)
        rmse = np.median(np.abs(ground_truth_es - model_es))

    residualall.append(residuals)
    rmseall.append(rmse)

    for fname in INTER_MODEL_FILES:
        model_strucs = read(str(DATA_DIR / "inter_MoS2WSe2" / fname), ":", format="extxyz")
        model_es = np.array(
            [
                get_property(property_name, atom=at)
                for at in model_strucs
                if len(at) == 12
            ]
        )
        if energy_prop:
            model_es = model_es - model_es[min_e_struc]

        if len(ground_truth_es.shape) > 1:
            ndata = ground_truth_es.shape[0]
            residuals = np.linalg.norm(
                np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
            )
            rmse = np.median(
                np.linalg.norm(
                    np.abs(ground_truth_es - model_es).reshape(ndata, -1), axis=-1
                )
            )
        else:
            residuals = np.abs(ground_truth_es - model_es)
            rmse = np.median(np.abs(ground_truth_es - model_es))

        residualall.append(residuals)
        rmseall.append(rmse)

    data = np.array(residualall).T
    rmse_data = np.array(rmseall)
    return data, rmse_data


def run_plot(property_name, energy_prop, inter, ax, with_gradient=False):
    if not inter:
        colorall = colorintra
        hatchesall = [None, None, "++", "++", "xx"]
        data, rmse_data = get_residuals(property_name, energy_prop)
    else:
        colorall = colorinter
        hatchesall = [None, None, "xx", "++", "xx"]
        data, rmse_data = get_residuals_inter(property_name, energy_prop)

    if not inter:
        if property_name == "energy_per_atom":
            lower_accuracy = 1e-2
        elif property_name == "all_forces":
            lower_accuracy = 10
    else:
        if property_name == "energy_per_atom":
            lower_accuracy = 1e-2
        elif property_name == "all_forces":
            lower_accuracy = 10
    data = data + lower_accuracy
    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=0)

    if property_name == "energy_per_atom":
        if inter:
            desired_value = 1
            col = "#66c2a5"
        else:
            desired_value = 0.1 * 1000 / 6
            col = "#fc8d62"
    elif property_name == "all_forces":
        if inter:
            desired_value = 200
            col = "#66c2a5"
        else:
            desired_value = 4000
            col = "#fc8d62"

    width = 100
    height = 256
    gradient = generate_gradient(col, width, height)

    extent = [0.5, 6, desired_value * 1e-3, desired_value]

    if inter:
        if with_gradient:
            ax.imshow(gradient, aspect="auto", extent=extent)
        ax.hlines(desired_value, 0.5, 6, color=col, linestyle="-", linewidth=2, alpha=0.7)
        nplt = 5
    else:
        if with_gradient:
            ax.imshow(gradient, aspect="auto", extent=extent)
        ax.hlines(desired_value, 0.5, 6, color=col, linestyle="-", linewidth=2, alpha=0.7)
        nplt = 4

    parts = ax.violinplot(
        data[:, :nplt],
        showmeans=False,
        points=data.shape[0],
        showextrema=False,
        widths=0.3,
    )

    parts_2 = ax.violinplot(
        data[:, :nplt],
        showmeans=False,
        points=data.shape[0],
        showextrema=False,
        widths=0.3,
    )

    parts_3 = ax.violinplot(
        data[:, :nplt],
        showmeans=False,
        points=data.shape[0],
        showextrema=False,
        widths=0.3,
    )

    yerr = np.vstack(
        ((medians - quartile1).reshape(1, -1), (quartile3 - medians).reshape(1, -1))
    )
    lwuse = 1
    for i in range(1, nplt + 1):
        ax.errorbar(
            i,
            medians[i - 1],
            yerr=yerr[:, i - 1].reshape(-1, 1),
            c="k",
            capthick=lwuse + 1,
            capsize=7.5,
            lw=lwuse + 0.5,
            zorder=5,
        )
        ax.errorbar(
            i,
            medians[i - 1],
            yerr=yerr[:, i - 1].reshape(-1, 1),
            c=colorall[i - 1],
            capthick=lwuse,
            lw=lwuse,
            capsize=7,
            zorder=6,
        )

    for ct, pc in enumerate(parts["bodies"]):
        pc.set_alpha(0.4)
        pc.set_facecolor(colorall[ct])
        pc.set_zorder(4)
        pc.set_edgecolor("none")

    for ct, pc in enumerate(parts_2["bodies"]):
        pc.set_alpha(1)
        pc.set_facecolor("none")
        pc.set_hatch(hatchesall[ct])
        pc.set_edgecolor(colorall[ct])
        pc.set_linewidth(lwuse)
        pc.set_zorder(3)

    for ct, pc in enumerate(parts_3["bodies"]):
        pc.set_alpha(1)
        pc.set_facecolor("none")
        pc.set_edgecolor("k")
        pc.set_hatch(hatchesall[ct])
        pc.set_linewidth(lwuse + 0.5)
        pc.set_zorder(-1)

    ax.set_ylabel(f"{get_property_legend(property_name)}")
    ax.set_yscale("log")
    if property_name == "energy_per_atom":
        ax.set_ylim(8e-3, 1000)
        ax.set_yticks(
            [1e-2, 1e-1, 1, 10, 100, 1000],
            labels=[r"≤10⁻²", 0.1, 1, 10, r"100", r"10³"],
        )
    elif property_name == "all_forces":
        ax.set_ylim(lower_accuracy, 50000)
        ax.set_yticks(
            [10, 100, 1000, 10000],
            labels=[r"≤10", r"10²", r"10³", r"10⁴"],
        )

    ax.scatter(np.arange(1, nplt + 1), medians[:nplt], marker="o", c="k", s=20, zorder=7)
    ax.scatter(
        np.arange(1, nplt + 1),
        medians[:nplt],
        marker="o",
        c=colorall[:nplt],
        s=10,
        zorder=8,
    )

    if not inter:
        ax.set_xticks(
            [1, 2, 3, 4, 5],
            ["", "", "Unified\nModel", "Split\nModel", "Unified\nModel"],
        )
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
            label.set_multialignment("center")

        sec = ax.secondary_xaxis("bottom")
        sec.set_xticks(
            [1, 2, 3.5, 5],
            labels=[
                "Random\nGuessing",
                "Stillinger\nWeber",
                "\n\n" r"$r_{cut}$= 6.0 Å",
                "\n\n" r"$r_{cut}$= 10.0 Å",
            ],
        )
        sec.tick_params("x", length=0)
        sec2 = ax.secondary_xaxis("bottom")
        sec2.set_xticks([0.5, 1.5, 2.5, 4.5, 5.5], labels=[])
        sec2.tick_params("x", length=25, direction="out", width=1)
        ax.set_xlim(0.5, 4.5)
    else:
        ax.set_xticks(
            [1, 2, 3, 4, 5],
            ["", "", "", "\nSmall\nNN", "\nLarge\nNN"],
            multialignment="center",
        )
        sec = ax.secondary_xaxis("bottom")
        sec.set_xticks(
            [1, 2, 3, 4.5],
            labels=["Random\nGuessing", "KC", "Unified\nModel", "Split Model"],
            multialignment="center",
        )
        sec.tick_params("x", length=0)
        sec2 = ax.secondary_xaxis(location=0)
        sec2.set_xticks([0.5, 1.5, 2.5, 3.5, 5.5], labels=[])
        sec2.tick_params("x", length=25, direction="out", width=1)
        ax.set_xlim(0.5, 5.5)


if __name__ == "__main__":
    # Load custom fonts if available
    home_dir = os.environ.get("HOME", "")
    fnt_pths = os.path.join(home_dir, "fonts")
    if os.path.isdir(fnt_pths):
        fnts_files = fm.findSystemFonts(fontpaths=fnt_pths, fontext="ttf")
        for fnt in fnts_files:
            fm.fontManager.addfont(fnt)

    rc_path = SCRIPT_DIR / "matplotlib.rc"
    if rc_path.exists():
        plt.style.use(str(rc_path))

    np.set_printoptions(precision=2)

    property_names = ["energy_per_atom", "energy_per_atom", "all_forces", "all_forces"]
    energy_props = [True, True, False, False]
    inters = [False, True, False, True]

    fig, allaxs = plt.subplots(
        2,
        1,
        figsize=(FIGWIDTH, FIGHEIGHT),
        layout="constrained",
    )
    axs = allaxs.flatten()

    for ct in range(0, 2):
        run_plot(property_names[ct], energy_props[ct], inters[ct], axs[ct], True)

    for i in [0, 1]:
        axs[i].tick_params(axis="y", which="major", pad=2)

    fig.align_labels()
    fig.savefig(SCRIPT_DIR / "fig_2.pdf")
    fig.savefig(SCRIPT_DIR / "fig_2.png")
