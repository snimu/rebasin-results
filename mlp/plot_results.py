from __future__ import annotations

import ast
import copy
import math
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams['text.usetex'] = True  # use latex for font rendering


def get_weight_decay(name: str) -> float:
    return float(name.split("wd")[1].split("_")[0])


def plot_pcd_results() -> None:
    fig, axs = plt.subplots(11, 2, figsize=(10, 19))
    fig.subplots_adjust(top=0.94, bottom=0.06)

    fig.suptitle("Permutation Coordinate Descent: MLP")

    loss_files = []
    acc_files = []

    for file in os.listdir("results/permutation-coordinate-descent"):
        if file.startswith("loss"):
            loss_files.append(file)
        elif file.startswith("acc"):
            acc_files.append(file)

    loss_files.sort(key=get_weight_decay)
    acc_files.sort(key=get_weight_decay)

    for i, (loss_file, acc_file) in enumerate(zip(loss_files, acc_files)):
        loss_df = pd.read_csv(os.path.join("results/permutation-coordinate-descent", loss_file))
        acc_df = pd.read_csv(os.path.join("results/permutation-coordinate-descent", acc_file))

        axs[i, 0].plot(loss_df["a-b-orig"].values, color="blue")
        axs[i, 0].plot(loss_df["a-b-rebasin"].values, color="red")
        axs[i, 0].plot(loss_df["b-orig-b-rebasin"].values, color="green")
        axs[i, 0].set_title(f"weight decay: {get_weight_decay(loss_file)}")
        axs[i, 0].set_ylabel("Loss")
        axs[i, 0].grid()

        axs[i, 1].plot(acc_df["a-b-orig"].values, color="blue")
        axs[i, 1].plot(acc_df["a-b-rebasin"].values, color="red")
        axs[i, 1].plot(acc_df["b-orig-b-rebasin"].values, color="green")
        axs[i, 1].set_title(f"weight decay: {get_weight_decay(acc_file)}")
        axs[i, 1].set_ylabel("Accuracy")
        axs[i, 1].grid()

        if i < len(loss_files) - 1:
            axs[i, 0].set_xticklabels([])
            axs[i, 1].set_xticklabels([])

    axs[-1, 0].set_xlabel("Interpolation %")
    axs[-1, 1].set_xlabel("Interpolation %")

    fig.legend(
        labels=["a-b-orig", "a-b-rebasin", "b-orig-b-rebasin"],
        loc="lower center",
        ncol=3,
        labelcolor=["blue", "red", "green"],
    )

    plt.savefig("results/permutation-coordinate-descent/pcd_results.png")


def plot_mm_hf() -> None:
    results_2_100 = pd.read_csv("results/merge-many/results_hf_2-100.csv")
    savefile = "results/merge-many/merge_many_hf_2-100.png"
    xkey = "hidden_features"
    xaxislabel = "Hidden Features"
    title = "MergeMany: weight_decay=0.0, num_models=3"

    plot_one_mm_plot(results_2_100, title, savefile, xkey, xaxislabel)

    results_2_100 = normalize(results_2_100, "loss_merged", "acc_merged")
    savefile = "results/merge-many/merge_many_hf_2-100_norm.png"
    title = "MergeMany: weight_decay=0.0, num_models=3 (normalized)"

    plot_one_mm_plot(results_2_100, title, savefile, xkey, xaxislabel)

    results_100_2000 = pd.read_csv("results/merge-many/results_hf_100-2000.csv")
    savefile = "results/merge-many/merge_many_hf_100-2000.png"
    xkey = "hidden_features"
    xaxislabel = "Hidden Features"
    title = "MergeMany: weight_decay=0.0, num_models=3"

    plot_one_mm_plot(results_100_2000, title, savefile, xkey, xaxislabel)

    results_100_2000 = normalize(results_100_2000, "loss_merged", "acc_merged")
    savefile = "results/merge-many/merge_many_hf_100-2000_norm.png"
    title = "MergeMany: weight_decay=0.0, num_models=3 (normalized)"

    plot_one_mm_plot(results_100_2000, title, savefile, xkey, xaxislabel)


def plot_mm_wd() -> None:
    results_lin = pd.read_csv("results/merge-many/results_wd_0.0-0.1.csv")
    savefile = "results/merge-many/merge_many_wd_0.0-0.1.png"
    xkey = "weight_decay"
    xaxislabel = "Weight Decay"
    title = "MergeMany: hidden_features=100, num_models=3"

    plot_one_mm_plot(results_lin, title, savefile, xkey, xaxislabel)

    results_lin = normalize(results_lin, "loss_merged", "acc_merged")
    savefile = "results/merge-many/merge_many_wd_0.0-0.1_norm.png"
    title = "MergeMany: hidden_features=100, num_models=3 (normalized)"

    plot_one_mm_plot(results_lin, title, savefile, xkey, xaxislabel)

    results_log = pd.read_csv("results/merge-many/results_wd_0.0-0.1_log.csv")
    savefile = "results/merge-many/merge_many_wd_0.0-0.1_log.png"
    title = "MergeMany: hidden_features=100, num_models=3"

    plot_one_mm_plot(results_log, title, savefile, xkey, xaxislabel)

    results_log = normalize(results_log, "loss_merged", "acc_merged")
    savefile = "results/merge-many/merge_many_wd_0.0-0.1_log_norm.png"
    title = "MergeMany: hidden_features=100, num_models=3 (normalized)"

    plot_one_mm_plot(results_log, title, savefile, xkey, xaxislabel)

    results_large = pd.read_csv("results/merge-many/results_wd_0.0-0.9.csv")
    savefile = "results/merge-many/merge_many_wd_0.0-0.9.png"
    title = "MergeMany: hidden_features=100, num_models=3"

    plot_one_mm_plot(results_large, title, savefile, xkey, xaxislabel)

    results_large = normalize(results_large, "loss_merged", "acc_merged")
    savefile = "results/merge-many/merge_many_wd_0.0-0.9_norm.png"
    title = "MergeMany: hidden_features=100, num_models=3 (normalized)"

    plot_one_mm_plot(results_large, title, savefile, xkey, xaxislabel)


def plot_one_mm_plot(
        results: pd.DataFrame, title: str, savefile: str, xkey: str, xaxislabel: str
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig.subplots_adjust(top=0.9, bottom=0.17)

    fig.suptitle(title)

    axs[0].plot(
        results[xkey],
        results["loss_avg"],
        label="loss avg", color="blue"
    )
    axs[0].plot(
        results[xkey],
        results["loss_merged"],
        label="loss merged", color="red"
    )
    axs[0].set_title("Loss")
    axs[0].set_ylabel("Loss")
    axs[0].grid()

    axs[1].plot(
        results[xkey],
        results["acc_avg"],
        label="acc avg", color="blue"
    )
    axs[1].plot(
        results[xkey],
        results["acc_merged"],
        label="acc merged", color="red"
    )
    axs[1].set_title("Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].grid()

    axs[0].set_xlabel(xaxislabel)
    axs[1].set_xlabel(xaxislabel)

    fig.legend(
        labels=["avg model", "merged model"],
        loc="lower center",
        ncol=2,
        labelcolor=["blue", "red"],
    )

    # Save here
    fig.savefig(savefile, dpi=300)
    plt.cla()
    plt.clf()
    plt.close(fig)


def plot_full_wd_hf_sweep() -> None:
    results = pd.read_csv("results/merge-many/full_wd0.0-0.2_hf100-2000_sweep.csv")

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(top=0.9, bottom=0.15, hspace=0.35, wspace=0.25, left=0.1, right=0.95)
    fig.suptitle("MergeMany: num_models=3")

    hidden_features = results["hidden_features"].unique()

    for hf in hidden_features:
        results_hf = results[results["hidden_features"] == hf]

        loss_ratio = results_hf["loss_merged"] / results_hf["loss_avg"]
        acc_ratio = results_hf["acc_merged"] / results_hf["acc_avg"]

        lratio_text = f"loss ratio (merged/avg)"
        aratio_text = f"acc ratio (merged/avg)"

        axs[0, 0].plot(
            results_hf["weight_decay"],
            loss_ratio,
            label=f"hidden_features={hf}"
        )
        axs[0, 0].set_ylabel(lratio_text)

        axs[0, 1].plot(
            results_hf["weight_decay"],
            acc_ratio,
        )
        axs[0, 1].set_ylabel(aratio_text)

        axs[1, 0].plot(
            results_hf["weight_decay"],
            loss_ratio,
        )
        axs[1, 0].set_ylabel(lratio_text)

        axs[1, 1].plot(
            results_hf["weight_decay"],
            acc_ratio,
        )
        axs[1, 1].set_ylabel(aratio_text)

    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()

    axs[0, 0].set_title("Loss Ratio")
    axs[0, 1].set_title("Accuracy Ratio")
    axs[1, 0].set_title("Loss Ratio (zoomed)")
    axs[1, 1].set_title("Accuracy Ratio (zoomed)")

    axs[1, 0].set_ylim([0.995, 1.05])
    axs[1, 1].set_ylim([0.95, 1.05])

    axs[0, 0].set_xlabel("Weight Decay")
    axs[0, 1].set_xlabel("Weight Decay")
    axs[1, 0].set_xlabel("Weight Decay")
    axs[1, 1].set_xlabel("Weight Decay")
    fig.legend(
        ncol=3,
        loc="lower center",
    )

    savefile = "results/merge-many/full_wd0.0-0.2_hf100-2000_sweep.png"
    fig.savefig(savefile, dpi=300)


def plot_compare_output_statistics() -> None:
    results = pd.read_csv("results/merge-many/compare_output_statistics_wd0.0-0.2_hf400.csv")

    plt.suptitle("MergeMany: num_models=3, hidden_features=400")

    plt.plot(
        results["weight_decay"],
        results["loss_ratio"],
        label="loss ratio (merged/avg)"
    )
    plt.plot(
        results["weight_decay"],
        results["acc_ratio"],
        label="acc ratio (merged/avg)"
    )
    plt.plot(
        results["weight_decay"],
        results["max_merged"] / results["max_avg"],
        label="max ratio (merged/avg)"
    )
    plt.plot(
        results["weight_decay"],
        results["std_merged"] / results["std_avg"],
        label="std ratio (merged/avg)"
    )
    plt.grid()
    plt.ylabel("metric ratio (merged/avg)")
    plt.xlabel("Weight Decay")

    plt.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
    )

    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.subplots_adjust(top=0.92, bottom=0.2, hspace=0.35, wspace=0.25, left=0.1, right=0.95)

    plt.savefig("results/merge-many/compare_output_statistics_wd0.0-0.2_hf400.png", dpi=300)


def plot_output_statistics_heatmap() -> None:
    results = pd.read_csv("results/merge-many/compare_output_statistics_wd0.0-0.2_hf50-1000_nm2-6.csv")

    results["max_ratio"] = results["max_merged"] / results["max_avg"]
    results["std_ratio"] = results["std_merged"] / results["std_avg"]
    num_models = results["num_models"].unique()

    fig, axs = plt.subplots(nrows=len(num_models), ncols=4, figsize=(13, 19))
    fig.subplots_adjust(top=2.2, hspace=3)

    for y, nm in enumerate(num_models):
        draw_heatmap(
            results, fig, axs, 0, y, nm, "viridis",
            f"Std Dev Ratio at {nm} models",
            r'$\frac{\mathtt{std}_{\mathrm{merged}}}{\mathtt{std}_{\mathrm{avg}}}$',
            "std_ratio",
            vmin=0.48,
            vmax=1.01,
        )
        draw_heatmap(
            results, fig, axs, 1, y, nm, "viridis",
            f"Max Ratio at {nm} models",
            r'$\frac{\mathtt{max}_{\mathrm{merged}}}{\mathtt{max}_{\mathrm{avg}}}$',
            "max_ratio",
            vmin=0.83,
            vmax=1.01,
        )
        draw_heatmap(
            results, fig, axs, 2, y, nm, "viridis",
            f"Loss Ratio at {nm} models",
            r'$\frac{\mathtt{loss}_{\mathrm{merged}}}{\mathtt{loss}_{\mathrm{avg}}}$',
            "loss_ratio",
            vmin=0.99,
            vmax=1.38,
        )
        draw_heatmap(
            results, fig, axs, 3, y, nm, "viridis",
            f"Acc Ratio at {nm} models",
            r'$\frac{\mathtt{acc}_{\mathrm{merged}}}{\mathtt{acc}_{\mathrm{avg}}}$',
            "acc_ratio",
            vmin=0.58,
            vmax=1.01,
        )

    fig.tight_layout()
    plt.savefig(
        "results/merge-many/compare_output_statistics_wd0.0-0.2_hf50-1000_nm2-6_heatmap.png",
        dpi=300
    )


def draw_heatmap(
        data: pd.DataFrame,
        fig,
        axs,
        x: int,
        y: int,
        num_models: int,
        cmap: str,
        title: str,
        label: str,
        feature: str,
        vmin: float,
        vmax: float,
) -> None:
    working_data = data[data["num_models"] == num_models]
    weight_decays = working_data["weight_decay"].unique()
    hidden_features = working_data["hidden_features"].unique()
    grid = np.zeros((len(hidden_features), len(weight_decays)))

    for i, wd in enumerate(weight_decays):
        for j, hf in enumerate(hidden_features):
            thisdata = (
                working_data[
                    (working_data["weight_decay"] == wd) & (working_data["hidden_features"] == hf)
                ]
            )
            grid[j, i] = thisdata[feature].values

    cax = axs[y, x].imshow(grid, cmap=cmap, origin='lower')
    axs[y, x].set_xticks(np.arange(len(weight_decays)/2)*2)
    axs[y, x].set_xticklabels(weight_decays[::2])
    axs[y, x].set_yticks(np.arange(len(hidden_features)))
    axs[y, x].set_yticklabels(hidden_features)
    axs[y, x].set_xlabel('Weight Decay')
    axs[y, x].set_ylabel('Hidden Features')
    axs[y, x].set_title(title)

    cax.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(cax, label=label)


def plot_count_permutations() -> None:
    results_02 = pd.read_csv("results/merge-many/count_permutations_wd0.0-0.2_hf400.csv")
    results_002 = pd.read_csv("results/merge-many/count_permutations_wd0.0-0.02_hf400.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.15, wspace=0.05, left=0.1, right=0.95)
    fig.suptitle("MergeMany: num_models=3, hidden_features=400")

    axs[0, 0].plot(
        results_002["weight_decay"],
        results_002["num_permutations"] / 1_000,
    )
    axs[0, 0].set_ylabel("# Permutations (in 1000)")
    axs[0, 0].set_ylim([680, 800])
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_xticks(np.arange(0.0, 0.021, 0.002))
    axs[0, 0].set_title("Weight Decay in [0.0, 0.02]")
    axs[0, 0].grid()

    axs[0, 1].plot(
        results_02["weight_decay"],
        results_02["num_permutations"] / 1_000,
    )
    axs[0, 1].set_ylim([680, 800])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xticks(np.arange(0.0, 0.21, 0.02))
    axs[0, 1].set_title("Weight Decay in [0.0, 0.2]")
    axs[0, 1].grid()

    axs[1, 0].plot(
        results_002["weight_decay"],
        results_002["num_permutations"] / 1_000,
    )
    axs[1, 0].set_ylabel("# Permutations (in 1000)")
    axs[1, 0].set_ylim([0, 800])
    axs[1, 0].set_xticks(np.arange(0.0, 0.021, 0.002))
    axs[1, 0].set_xlabel("Weight Decay")
    axs[1, 0].grid()

    axs[1, 1].plot(
        results_02["weight_decay"],
        results_02["num_permutations"] / 1_000,
    )
    axs[1, 1].set_ylim([0, 800])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_xticks(np.arange(0.0, 0.21, 0.02))
    axs[1, 1].set_xlabel("Weight Decay")
    axs[1, 1].grid()

    # plt.show()
    plt.savefig("results/merge-many/count_permutations.png", dpi=300)


def normalize(data: pd.DataFrame, key_loss: str, key_acc: str) -> pd.DataFrame:
    ndata = copy.deepcopy(data)
    norm_factors_loss = ndata[key_loss].values
    norm_factors_acc = ndata[key_acc].values
    ndata["loss_avg"] = ndata["loss_avg"] / norm_factors_loss
    ndata["loss_merged"] = ndata["loss_merged"] / norm_factors_loss
    ndata["acc_avg"] = ndata["acc_avg"] / norm_factors_acc
    ndata["acc_merged"] = ndata["acc_merged"] / norm_factors_acc
    return ndata


def plot_abs_weight_mean_diff() -> None:
    data = pd.read_csv("results/other/test_squared_weight_mean_differences_hf20-1000_wd0.0-0.2.csv")
    wds = data["weight_decay"].unique()

    for wd in wds:
        data_wd = data[data["weight_decay"] == wd]
        plt.plot(data_wd["hidden_features"], data_wd["mean_perc_diff"] * 100, label=f"wd={wd}")

    plt.grid()
    plt.ylabel("difference in %")
    plt.xlabel("hidden features")
    plt.legend()

    plt.show()


def plot_abs_weight_mean_diff3d() -> None:
    # 1. Read the data using pandas
    df = pd.read_csv("results/other/test_squared_weight_mean_differences_hf20-1000_wd0.0-0.2.csv")

    # 2. Pivot the DataFrame
    pivot_df = df.pivot(index='weight_decay', columns='hidden_features', values='mean_perc_diff')

    # 3. Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of x, y values
    x = pivot_df.columns.values
    y = pivot_df.index.values
    x, y = np.meshgrid(x, y)

    # Get corresponding z values
    z = pivot_df.values

    # 3D surface plot
    surf = ax.plot_surface(x, y, z, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Weight Decay')
    ax.set_ylabel('Hidden Features')
    ax.set_zlabel('Mean Percentage Difference')
    ax.set_title('3D plot of Mean Percentage Difference')

    # Add a color bar and shrink it
    fig.colorbar(surf, shrink=0.6)

    plt.tight_layout()
    plt.show()

    plt.show()


def plot_abs_weight_mean_diff_heatmap() -> None:
    # Read the CSV file
    data = pd.read_csv('results/other/weight_statistics_hf20-1000_wd0.0-1.0_nm2-2.csv')

    # Set a colormap
    cmap = "viridis"

    # Create a Figure
    fig = plt.figure(figsize=(11, 13))
    mean_abs_title = (
        r'$\texttt{mean}_{\mathrm{abs}} '
        r'= \texttt{mean}_i\left('
        r'\texttt{mean}_j\left( w_{am,i}^{m_j} \right)'
        r'\right)$'
    )
    mean_delta_title = (
        r'$\texttt{mean}_{\Delta\%} '
        r'= 100 \cdot \texttt{mean}_i\left('
        r'\frac{'
        r'\texttt{max}\left( w_{am,i}^{m_1}, w_{am,i}^{m_2} \right) '
        r'- \texttt{min}\left( w_{am,i}^{m_1}, w_{am,i}^{m_2} \right)'
        r'}{\texttt{mean}\left(w_{am,i}^{m_1}, w_{am,i}^{m_2}\right)}'
        r'\right)$'
    )
    eigval_delta_title = (
        r'$\texttt{eigval}_{\Delta\%} '
        r'= 100 \cdot \texttt{mean}_i\left('
        r'\frac{'
        r'\left| \texttt{eigval}_i^{m_1} - \texttt{eigval}_i^{m_2} \right|'
        r'}{\texttt{mean}\left( \texttt{eigval}_i^{m_1}, \texttt{eigval}_i^{m_2} \right)}'
        r'\right)$'
    )
    eigvec_angle_title = (
        r'$\texttt{eigvec}_{\mathrm{angle}} '
        r'= \texttt{mean}_j\left('
        r'\texttt{mean}_k\left('
        r'\texttt{angle}\left(\texttt{eigvec}^{k,m_j}, \texttt{eigvec}^{k+1,m_j}\right)'
        r'\right)'
        r'\right)$'
    )
    fig.suptitle(
        r'\textbf{Weight Statistics (for 2 models)} \\---\\ '
        r'$w_{am,i}^{model_j} = \texttt{mean}\left(\texttt{abs}\left(w_i^{m_j}\right)\right)$, '
        r'$w$: weight; $m$: model; $i$: weight-index; $j$: model-index'
        r'\\---\\'
        r'$\texttt{eigval}_i^k$: $k^{th}$ eigenvalue of $w_i$; $\texttt{eigvec}_i^k$: $k^{th}$ eigenvector of $w_i$'
        r' \\---\\ ' + mean_abs_title +
        r' \\---\\ ' + mean_delta_title +
        r' \\---\\ ' + eigval_delta_title +
        r' \\---\\ ' + eigvec_angle_title
    )
    fig.subplots_adjust(top=0.75, bottom=0.05, left=0.05, right=0.95, hspace=0.2, wspace=0.2)

    # Extract unique values of weight_decay and hidden_features
    weight_decays = data['weight_decay'].unique()
    hidden_features = data['hidden_features'].unique()
    weight_decays.sort()
    hidden_features.sort()

    # Create a grid of mean_perc_diff values
    grid = np.zeros((len(hidden_features), len(weight_decays)))

    # abs_mean_val
    for i, weight_decay in enumerate(weight_decays):
        for j, hidden_feature in enumerate(hidden_features):
            abs_mean_val = data[(data['weight_decay'] == weight_decay) & (data['hidden_features'] == hidden_feature)][
                'abs_mean_val']
            grid[j, i] = np.log(abs_mean_val.values[0])

    ax1 = fig.add_subplot(221)
    cax1 = ax1.imshow(grid, cmap=cmap, origin='lower')
    ax1.set_xticks(np.arange(len(weight_decays)/4)*4)
    ax1.set_xticklabels(weight_decays[::4])
    ax1.set_yticks(np.arange(len(hidden_features)/2)*2)
    ax1.set_yticklabels(hidden_features[::2])
    ax1.set_xlabel('Weight Decay')
    ax1.set_ylabel('Hidden Features')
    ax1.set_title(r'$\texttt{log} \left( \texttt{mean}_{\mathrm{abs}} \right)$')

    fig.colorbar(cax1, label=r'$\texttt{log} \left( \texttt{mean}_{\mathrm{abs}} \right)$')

    # abs_mean_diff
    for i, weight_decay in enumerate(weight_decays):
        for j, hidden_feature in enumerate(hidden_features):
            abs_mean_diff = data[(data['weight_decay'] == weight_decay) & (data['hidden_features'] == hidden_feature)][
                'abs_mean_diff']
            grid[j, i] = abs_mean_diff.values[0] * 100

    ax2 = fig.add_subplot(222)
    cax2 = ax2.imshow(grid, cmap=cmap, origin='lower')
    ax2.set_xticks(np.arange(len(weight_decays)/4)*4)
    ax2.set_xticklabels(weight_decays[::4])
    ax2.set_yticks(np.arange(len(hidden_features)/2)*2)
    ax2.set_yticklabels(hidden_features[::2])
    ax2.set_xlabel('Weight Decay')
    ax2.set_ylabel('Hidden Features')
    ax2.set_title(r'$\texttt{mean}_{\Delta\%}$')

    fig.colorbar(cax2, label=r'$\texttt{mean}_{\Delta\%}$ $[\%]$')

    # eigvals
    for i, weight_decay in enumerate(weight_decays):
        for j, hidden_feature in enumerate(hidden_features):
            eigvals = data[(data['weight_decay'] == weight_decay) & (data['hidden_features'] == hidden_feature)][
                'eigval_diff']
            grid[j, i] = eigvals.values[0] * 100

    ax3 = fig.add_subplot(223)
    cax3 = ax3.imshow(grid, cmap=cmap, origin='lower')
    ax3.set_xticks(np.arange(len(weight_decays)/4)*4)
    ax3.set_xticklabels(weight_decays[::4])
    ax3.set_yticks(np.arange(len(hidden_features)/2)*2)
    ax3.set_yticklabels(hidden_features[::2])
    ax3.set_xlabel('Weight Decay')
    ax3.set_ylabel('Hidden Features')
    ax3.set_title(r'$\texttt{eigval}_{\Delta\%}$')

    fig.colorbar(cax3, label=r'$\texttt{eigval}_{\Delta\%}$ $[\%]$')

    # eigvec_angle
    for i, weight_decay in enumerate(weight_decays):
        for j, hidden_feature in enumerate(hidden_features):
            eigvec_angle = data[(data['weight_decay'] == weight_decay) & (data['hidden_features'] == hidden_feature)][
                'eigvec_angle']
            grid[j, i] = eigvec_angle.values[0]

    ax4 = fig.add_subplot(224)
    cax4 = ax4.imshow(grid, cmap=cmap, origin='lower')
    ax4.set_xticks(np.arange(len(weight_decays)/4)*4)
    ax4.set_xticklabels(weight_decays[::4])
    ax4.set_yticks(np.arange(len(hidden_features)/2)*2)
    ax4.set_yticklabels(hidden_features[::2])
    ax4.set_xlabel('Weight Decay')
    ax4.set_ylabel('Hidden Features')
    ax4.set_title(r'$\texttt{eigvec}_{\mathrm{angle}}$')

    fig.colorbar(cax4, label=r'$\texttt{eigvec}_{\mathrm{angle}}$ $[\deg]$')

    plt.savefig(
        "results/other/weight_statistics_hf20-1000_wd0.0-1.0_nm2-2.png",
        dpi=300,
    )


def plot_mm_nm() -> None:
    df = pd.read_csv("results/merge-many/merge_many_nm2-6_wd0-0.3_nf200.csv")
    nms = df["num_models"].unique()

    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.subplots_adjust(top=0.9, bottom=0.2, hspace=0.5, wspace=0.3)
    fig.suptitle("Merge Many: different numbers of models (at 200 Hidden Features)")

    for nm in nms:
        df_nm = df[df["num_models"] == nm]
        axs[0, 0].plot(df_nm["weight_decay"], df_nm["loss_merged"] / df_nm["loss_avg"], label=f"{nm} models")
        axs[0, 0].set_ylabel("loss ratio (merged / avg)")
        axs[0, 0].set_xlabel("weight decay")
        axs[0, 0].grid()
        axs[0, 0].set_title("Loss Ratio")

        axs[0, 1].plot(df_nm["weight_decay"], df_nm["acc_merged"] / df_nm["acc_avg"])
        axs[0, 1].set_ylabel("acc ratio (merged / avg)")
        axs[0, 1].set_xlabel("weight decay")
        axs[0, 1].grid()
        axs[0, 1].set_title("Accuracy Ratio")

        axs[1, 0].plot(df_nm["weight_decay"], df_nm["loss_merged"] / df_nm["loss_avg"])
        axs[1, 0].set_ylabel("loss ratio (merged / avg)")
        axs[1, 0].set_xlabel("weight decay")
        axs[1, 0].grid()
        axs[1, 0].set_title("Loss Ratio (zoomed)")
        axs[1, 0].set_ylim([1.0, 1.05])

        axs[1, 1].plot(df_nm["weight_decay"], df_nm["acc_merged"] / df_nm["acc_avg"])
        axs[1, 1].set_ylabel("acc ratio (merged / avg)")
        axs[1, 1].set_xlabel("weight decay")
        axs[1, 1].grid()
        axs[1, 1].set_title("Accuracy Ratio (zoomed)")
        axs[1, 1].set_ylim([0.98, 1.03])

    fig.legend(
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
    )

    plt.savefig("results/merge-many/merge_many_nm2-6_wd0-0.3_nf200.png", dpi=300)


def plot_eigvec_angles_different_distributions() -> None:
    df = pd.read_csv("results/other/eigvec_angles_different_distributions.csv")

    ksizes = df["ksize"].unique()
    ksizes.sort()
    sample_nums = df["sample_num"].unique()

    fig, axs = plt.subplots(len(sample_nums), 2, figsize=(10, 18))
    fig.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4, wspace=0.2, left=0.07, right=0.98)
    fig.suptitle("Eigenvector Angles for Different Distributions")

    for i, sample_num in enumerate(sample_nums):
        axs[i, 0].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "uniform")
            ]["eigvec_angle"],
            color="orange"
        )
        axs[i, 0].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "gaussian")
            ]["eigvec_angle"],
            color="purple"
        )
        axs[i, 0].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "orthogonal")
            ]["eigvec_angle"],
            color="red"
        )
        axs[i, 0].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "sparse0.5")
            ]["eigvec_angle"],
            color="green"
        )
        axs[i, 0].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "dirac")
            ]["eigvec_angle"],
            color="blue"
        )
        axs[i, 0].set_xlabel("Kernel Size")
        axs[i, 0].set_ylabel(r"$\texttt{eigvec}_{\mathrm{angle}}$ $[\deg]$")
        axs[i, 0].set_title(f"Sample Num: {sample_num}")
        axs[i, 0].grid()

        axs[i, 1].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "uniform")
            ]["eigvec_angle"],
            color="orange"
        )
        axs[i, 1].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "gaussian")
            ]["eigvec_angle"],
            color="purple"
        )
        axs[i, 1].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "orthogonal")
            ]["eigvec_angle"],
            color="red"
        )
        axs[i, 1].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "sparse0.5")
            ]["eigvec_angle"],
            color="green"
        )
        axs[i, 1].plot(
            ksizes,
            df[
                (df["sample_num"] == sample_num) & (df["distribution"] == "dirac")
            ]["eigvec_angle"],
            color="blue"
        )
        axs[i, 1].set_xlabel("Kernel Size")
        axs[i, 1].set_ylabel(r"$\texttt{eigvec}_{\mathrm{angle}}$ $[\deg]$")
        axs[i, 1].set_title(f"Sample Num: {sample_num} (zoomed)")
        axs[i, 1].grid()
        axs[i, 1].set_ylim([89.6, 90.2])

    fig.legend(
        ncol=5,
        loc="lower center",
        #bbox_to_anchor=(0.5, 0.05),
        labelcolor=["orange", "purple", "red", "green", "blue"],
        labels=["uniform", "gaussian", "orthogonal", "sparse (s=0.5)", "dirac"],
    )

    # plt.show()
    plt.savefig("results/other/eigvec_angles_different_distributions.png", dpi=300)


def plot_histograms() -> None:
    df = pd.read_csv("results/other/weight_histograms_wd0.0-0.6_hf20-220_nbins100.csv")

    weight_decays = df["weight_decay"].unique()
    weight_decays.sort()
    hidden_features = df["hidden_features"].unique()
    hidden_features.sort()

    fig, axs = plt.subplots(len(hidden_features), len(weight_decays), figsize=(10, 12))
    fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.6, wspace=0.5, left=0.07, right=0.98)
    fig.suptitle("Histograms of weight-element values")

    for i, hidden_feature in enumerate(hidden_features):
        for j, weight_decay in enumerate(weight_decays):
            this_df = df[
                (df["hidden_features"] == hidden_feature) &
                (df["weight_decay"] == weight_decay)
            ]
            bins = ast.literal_eval(this_df["bins"].values[0])
            values1 = ast.literal_eval(this_df["values1"].values[0])
            values2 = ast.literal_eval(this_df["values2"].values[0])
            assert len(values1) == len(values2)
            values_overlap = [min(v1, v2) for v1, v2 in zip(values1, values2)]

            axs[i, j].bar(np.arange(len(bins)), values1, width=1, color="orange")
            axs[i, j].bar(np.arange(len(bins)), values2, width=1, color="purple")
            axs[i, j].bar(np.arange(len(bins)), values_overlap, width=1, color="black")
            xticks = np.array([0, int(len(bins) / 2), len(bins) - 1])
            xticklabels = [round(bins[int(i)], 2) for i in xticks]
            axs[i, j].set_xticks(xticks)
            axs[i, j].set_xticklabels(xticklabels)
            axs[i, j].set_title(f"hf: {hidden_feature}, wd: {weight_decay}")
            axs[i, j].set_xlabel("Magnitude")
            axs[i, j].set_ylabel("Element Count")

    fig.legend(
        ncol=3,
        loc="lower center",
        labelcolor=["orange", "purple", "black"],
        labels=["model 1", "model 2", "overlap"],
    )

    # plt.show()
    plt.savefig("results/other/weight_histograms_wd0.0-0.6_hf20-220_nbins100.png", dpi=300)


def draw_ratio(
        df: pd.DataFrame,
        grid_max: np.ndarray[Any, Any],
        grid_min: np.ndarray[Any, Any],
        key: str,
        i: int,
        j: int,
) -> None:
    metriclist = ast.literal_eval(df[key].values[0])
    ratio_max = max(metriclist) / max(metriclist[0], metriclist[-1])
    ratio_min = min(metriclist) / min(metriclist[0], metriclist[-1])
    grid_max[j, i] = ratio_max
    grid_min[j, i] = ratio_min


def plot_pcd_on_split_dataset_heatmap() -> None:
    df = pd.read_csv(
        "results/permutation-coordinate-descent/pcd_hf200-200_wd0.0-0.5_nl2-10_epochs2.csv"
    )
    wds = df["weight_decay"].unique()
    wds.sort()
    nls = df["num_layers"].unique()
    nls.sort()

    grid_l_ab_rebasin_max = np.zeros((len(nls), len(wds)))
    grid_l_ab_rebasin_min = np.zeros((len(nls), len(wds)))
    grid_a_ab_rebasin_max = np.zeros((len(nls), len(wds)))
    grid_a_ab_rebasin_min = np.zeros((len(nls), len(wds)))
    grid_l_ab_orig_max = np.zeros((len(nls), len(wds)))
    grid_l_ab_orig_min = np.zeros((len(nls), len(wds)))
    grid_a_ab_orig_max = np.zeros((len(nls), len(wds)))
    grid_a_ab_orig_min = np.zeros((len(nls), len(wds)))

    fig = plt.figure(figsize=(10, 14))
    fig.subplots_adjust(top=0.8, bottom=0.02, hspace=0.3, wspace=0.2, left=0.05, right=0.98)
    # fig = plt.figure(figsize=(9, 18))
    # fig.subplots_adjust(top=0.8, bottom=0.02, hspace=0.3, wspace=0.2, left=0.05, right=0.95)
    fig.suptitle(
        r"$\mathcal{L}$: Loss, $\mathcal{A}$: Accuracy, $\mathcal{R}$: Ratio \\---\\"
        r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}} = "
        r"\frac"
        r"{\max\left(\mathrm{losses}\right)}"
        r"{\max\left(\mathcal{L}_{\mathrm{model}_a}, \mathcal{L}_{\mathrm{model}_b}\right)}$: "
        r"If $> 1.0$, the original models are not the worst models. \\---\\"
        r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}} = "
        r"\frac"
        r"{\min\left(\mathrm{losses}\right)}"
        r"{\min\left(\mathcal{L}_{\mathrm{model}_a}, \mathcal{L}_{\mathrm{model}_b}\right)}$: "
        r"If $< 1.0$, the original models are not the best models. \\---\\"
        r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}} = "
        r"\frac"
        r"{\max\left(\mathrm{accuracies}\right)}"
        r"{\max\left(\mathcal{A}_{\mathrm{model}_a}, \mathcal{A}_{\mathrm{model}_b}\right)}$: "
        r"If $> 1.0$, the original models are not the best models. \\---\\"
        r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}} = "
        r"\frac"
        r"{\min\left(\mathrm{accuracies}\right)}"
        r"{\min\left(\mathcal{A}_{\mathrm{model}_a}, \mathcal{A}_{\mathrm{model}_b}\right)}$: "
        r"If $< 1.0$, the original models are not the worst models. \\---\\"
        r"(a-b-rebasin): Interpolation between $\mathrm{model}_a$ and $\mathrm{model}_b$ "
        r"after rebasing \\"
        r"(a-b-orig): Interpolation between $\mathrm{model}_a$ and $\mathrm{model}_b$ "
        r"before rebasing \\"
        r"\texttt{hidden_features}: $200$x$200$ \\"
    )

    for i, wd in enumerate(wds):
        for j, nl in enumerate(nls):
            loc_df = df[(df['weight_decay'] == wd) & (df['num_layers'] == nl)]

            draw_ratio(loc_df, grid_l_ab_rebasin_max, grid_l_ab_rebasin_min, "loss-a-b-rebasin", i, j)
            draw_ratio(loc_df, grid_a_ab_rebasin_max, grid_a_ab_rebasin_min, "acc-a-b-rebasin", i, j)
            draw_ratio(loc_df, grid_l_ab_orig_max, grid_l_ab_orig_min, "loss-a-b-orig", i, j)
            draw_ratio(loc_df, grid_a_ab_orig_max, grid_a_ab_orig_min, "acc-a-b-orig", i, j)

    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax8 = fig.add_subplot(428)

    ax1.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$ (a-b-rebasin)")
    ax2.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$ (a-b-rebasin)")
    ax3.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$ (a-b-rebasin)")
    ax4.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$ (a-b-rebasin)")
    ax5.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$ (a-b-orig)")
    ax8.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$ (a-b-orig)")
    ax7.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$ (a-b-orig)")
    ax6.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$ (a-b-orig)")

    cmap = "viridis"
    cax1 = ax1.imshow(grid_l_ab_rebasin_max, cmap=cmap, origin='lower')
    cax2 = ax2.imshow(grid_a_ab_rebasin_min, cmap=cmap, origin='lower')
    cax3 = ax3.imshow(grid_l_ab_rebasin_min, cmap=cmap, origin='lower')
    cax4 = ax4.imshow(grid_a_ab_rebasin_max, cmap=cmap, origin='lower')
    cax5 = ax5.imshow(grid_l_ab_orig_max, cmap=cmap, origin='lower')
    cax6 = ax6.imshow(grid_a_ab_orig_min, cmap=cmap, origin='lower')
    cax7 = ax7.imshow(grid_l_ab_orig_min, cmap=cmap, origin='lower')
    cax8 = ax8.imshow(grid_a_ab_orig_max, cmap=cmap, origin='lower')

    for ax in (ax1, ax4, ax3, ax2, ax5, ax8, ax7, ax6):
        ax.set_xlabel('Weight Decay')
        ax.set_ylabel('Number of Layers')
        ax.set_xticks(np.arange(len(wds)))
        ax.set_xticklabels(wds)
        ax.set_yticks(np.arange(len(nls)))
        ax.set_yticklabels(nls)

    for caxL1, caxL2, caxA1, caxA2 in zip((cax1, cax5), (cax3, cax7), (cax4, cax8), (cax2, cax6)):
        fig.colorbar(caxL1, label=r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$")
        fig.colorbar(caxL2, label=r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$")
        fig.colorbar(caxA1, label=r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$")
        fig.colorbar(caxA2, label=r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$")

    for i, wd in enumerate(wds):
        for j, nl in enumerate(nls):
            ax1.text(i, j, round(grid_l_ab_rebasin_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax4.text(i, j, round(grid_a_ab_rebasin_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax3.text(i, j, round(grid_l_ab_rebasin_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax2.text(i, j, round(grid_a_ab_rebasin_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax5.text(i, j, round(grid_l_ab_orig_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax8.text(i, j, round(grid_a_ab_orig_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax7.text(i, j, round(grid_l_ab_orig_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax6.text(i, j, round(grid_a_ab_orig_min[j, i], 2),
                    ha="center", va="center", color="w")

    # plt.show()
    plt.savefig(
        "results/permutation-coordinate-descent/pcd_hf200-200_wd0.0-0.5_nl2-10_epochs2.png",
        dpi=300,
    )


def plot_pcd_on_split_dataset_heatmap_hf_nl() -> None:
    df = pd.read_csv(
        "results/permutation-coordinate-descent/pcd_hf100-1000_wd0.0-0.9_nl5-50_epochs1.csv"
    )
    df = df[df["weight_decay"] == 0.0]
    hfs = df["hidden_features"].unique()
    hfs.sort()
    nls = df["num_layers"].unique()
    nls.sort()

    grid_l_ab_rebasin_max = np.zeros((len(nls), len(hfs)))
    grid_l_ab_rebasin_min = np.zeros((len(nls), len(hfs)))
    grid_a_ab_rebasin_max = np.zeros((len(nls), len(hfs)))
    grid_a_ab_rebasin_min = np.zeros((len(nls), len(hfs)))
    grid_l_ab_orig_max = np.zeros((len(nls), len(hfs)))
    grid_l_ab_orig_min = np.zeros((len(nls), len(hfs)))
    grid_a_ab_orig_max = np.zeros((len(nls), len(hfs)))
    grid_a_ab_orig_min = np.zeros((len(nls), len(hfs)))

    fig = plt.figure(figsize=(9, 18))
    fig.subplots_adjust(top=0.8, bottom=0.04, hspace=0.3, wspace=0.2, left=0.05, right=0.95)
    fig.suptitle(
        r"$\mathcal{L}$: Loss, $\mathcal{A}$: Accuracy, $\mathcal{R}$: Ratio \\---\\"
        r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}} = "
        r"\frac"
        r"{\max\left(\mathrm{losses}\right)}"
        r"{\max\left(\mathcal{L}_{\mathrm{model}_a}, \mathcal{L}_{\mathrm{model}_b}\right)}$: "
        r"If $> 1.0$, the original models are not the worst models. \\---\\"
        r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}} = "
        r"\frac"
        r"{\min\left(\mathrm{losses}\right)}"
        r"{\min\left(\mathcal{L}_{\mathrm{model}_a}, \mathcal{L}_{\mathrm{model}_b}\right)}$: "
        r"If $< 1.0$, the original models are not the best models. \\---\\"
        r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}} = "
        r"\frac"
        r"{\max\left(\mathrm{accuracies}\right)}"
        r"{\max\left(\mathcal{A}_{\mathrm{model}_a}, \mathcal{A}_{\mathrm{model}_b}\right)}$: "
        r"If $> 1.0$, the original models are not the best models. \\---\\"
        r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}} = "
        r"\frac"
        r"{\min\left(\mathrm{accuracies}\right)}"
        r"{\min\left(\mathcal{A}_{\mathrm{model}_a}, \mathcal{A}_{\mathrm{model}_b}\right)}$: "
        r"If $< 1.0$, the original models are not the worst models. \\---\\"
        r"(a-b-rebasin): Interpolation between $\mathrm{model}_a$ and $\mathrm{model}_b$ "
        r"after rebasing \\"
        r"(a-b-orig): Interpolation between $\mathrm{model}_a$ and $\mathrm{model}_b$ "
        r"before rebasing \\"
        r"\texttt{weight_decay}: $0.0$ \\"
    )

    for i, hf in enumerate(hfs):
        for j, nl in enumerate(nls):
            loc_df = df[(df['hidden_features'] == hf) & (df['num_layers'] == nl)]

            draw_ratio(loc_df, grid_l_ab_rebasin_max, grid_l_ab_rebasin_min, "loss-a-b-rebasin", i, j)
            draw_ratio(loc_df, grid_a_ab_rebasin_max, grid_a_ab_rebasin_min, "acc-a-b-rebasin", i, j)
            draw_ratio(loc_df, grid_l_ab_orig_max, grid_l_ab_orig_min, "loss-a-b-orig", i, j)
            draw_ratio(loc_df, grid_a_ab_orig_max, grid_a_ab_orig_min, "acc-a-b-orig", i, j)

    ax1 = fig.add_subplot(421)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax8 = fig.add_subplot(428)

    ax1.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$ (a-b-rebasin)")
    ax2.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$ (a-b-rebasin)")
    ax3.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$ (a-b-rebasin)")
    ax4.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$ (a-b-rebasin)")
    ax5.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$ (a-b-orig)")
    ax7.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$ (a-b-orig)")
    ax6.set_title(r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$ (a-b-orig)")
    ax8.set_title(r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$ (a-b-orig)")

    cmap = "viridis"
    cax1 = ax1.imshow(grid_l_ab_rebasin_max, cmap=cmap, origin='lower')
    cax2 = ax2.imshow(grid_a_ab_rebasin_min, cmap=cmap, origin='lower')
    cax3 = ax3.imshow(grid_l_ab_rebasin_min, cmap=cmap, origin='lower')
    cax4 = ax4.imshow(grid_a_ab_rebasin_max, cmap=cmap, origin='lower')
    cax5 = ax5.imshow(grid_l_ab_orig_max, cmap=cmap, origin='lower')
    cax6 = ax6.imshow(grid_a_ab_orig_min, cmap=cmap, origin='lower')
    cax7 = ax7.imshow(grid_l_ab_orig_min, cmap=cmap, origin='lower')
    cax8 = ax8.imshow(grid_a_ab_orig_max, cmap=cmap, origin='lower')

    for ax in (ax1, ax2, ax3, ax4, ax5, ax8, ax7, ax6):
        ax.set_xlabel('Hidden Features')
        ax.set_ylabel('Number of Layers')
        ax.set_xticks(np.arange(len(hfs)))
        ax.set_xticklabels(hfs)
        ax.set_yticks(np.arange(len(nls)))
        ax.set_yticklabels(nls)

    for caxL1, caxL2, caxA1, caxA2 in zip((cax1, cax5), (cax3, cax7), (cax4, cax8), (cax2, cax6)):
        fig.colorbar(caxL1, label=r"$\mathcal{R}_{\mathrm{max}, \mathcal{L}}$")
        fig.colorbar(caxL2, label=r"$\mathcal{R}_{\mathrm{min}, \mathcal{L}}$")
        fig.colorbar(caxA1, label=r"$\mathcal{R}_{\mathrm{max}, \mathcal{A}}$")
        fig.colorbar(caxA2, label=r"$\mathcal{R}_{\mathrm{min}, \mathcal{A}}$")

    for i, hf in enumerate(hfs):
        for j, nl in enumerate(nls):
            ax1.text(i, j, round(grid_l_ab_rebasin_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax4.text(i, j, round(grid_a_ab_rebasin_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax3.text(i, j, round(grid_l_ab_rebasin_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax2.text(i, j, round(grid_a_ab_rebasin_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax5.text(i, j, round(grid_l_ab_orig_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax8.text(i, j, round(grid_a_ab_orig_max[j, i], 2),
                    ha="center", va="center", color="w")
            ax7.text(i, j, round(grid_l_ab_orig_min[j, i], 2),
                    ha="center", va="center", color="w")
            ax6.text(i, j, round(grid_a_ab_orig_min[j, i], 2),
                    ha="center", va="center", color="w")

    # plt.show()
    plt.savefig(
        "results/permutation-coordinate-descent/pcd_hf100-1000_wd0.0-0.0_nl5-50_epochs1.png",
        dpi=300,
    )


def plot_pcd_on_split_dataset_lineplot(wd: float, hf: int, nl: int) -> None:
    df = pd.read_csv(
        f"results/permutation-coordinate-descent/"
        f"pcd_hf200-200_wd0.0-0.9_nl2-20_epochs1.csv"
    )
    df = df[
        (df["weight_decay"] == wd)
        & (df["hidden_features"] == hf)
        & (df["num_layers"] == nl)
    ]
    loss_a_b_orig = ast.literal_eval(df["loss-a-b-orig"].values[0])
    loss_a_b_rebasin = ast.literal_eval(df["loss-a-b-rebasin"].values[0])
    loss_b_orig_b_rebasin = ast.literal_eval(df["loss-b-orig-b-rebasin"].values[0])
    acc_a_b_orig = ast.literal_eval(df["acc-a-b-orig"].values[0])
    acc_a_b_rebasin = ast.literal_eval(df["acc-a-b-rebasin"].values[0])
    acc_b_orig_b_rebasin = ast.literal_eval(df["acc-b-orig-b-rebasin"].values[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(
        f"PCD on Split Dataset "
        f"(weight_decay={wd}, hidden_features={hf}, num_layers={nl})"
    )
    fig.subplots_adjust(bottom=0.2, top=0.85)

    x = np.arange(len(loss_a_b_orig), dtype=np.float64) / len(loss_a_b_orig) * 100
    ax1.plot(x, loss_a_b_orig, label="a-b-orig")
    ax1.plot(x, loss_a_b_rebasin, label="a-b-rebasin")
    ax1.plot(x, loss_b_orig_b_rebasin, label="b-orig-b-rebasin")
    ax1.set_title("Loss")
    ax1.set_xlabel("Interpolation %")
    ax1.set_ylabel("Loss")

    ax2.plot(x, acc_a_b_orig)
    ax2.plot(x, acc_a_b_rebasin)
    ax2.plot(x, acc_b_orig_b_rebasin)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Interpolation %")
    ax2.set_ylabel("Accuracy")

    fig.legend(
        ncol=3,
        loc="lower center",
    )
    # plt.show()
    plt.savefig(
        f"results/permutation-coordinate-descent/pcd_hf{hf}_wd{wd}_nl{nl}_epochs1.png",
        dpi=300,
    )


if __name__ == "__main__":
    plot_pcd_on_split_dataset_lineplot(wd=0.1, nl=10, hf=200)
