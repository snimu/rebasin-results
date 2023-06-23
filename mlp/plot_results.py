from __future__ import annotations

import copy
import os

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
        r'\texttt{max}\left( \texttt{eigval}_i^{m_1}, \texttt{eigval}_i^{m_2} \right) '
        r'- \texttt{min}\left( \texttt{eigval}_i^{m_1}, \texttt{eigval}_i^{m_2} \right)'
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

    distributions = ("uniform", "gaussian", "dirac", "orthogonal", "sparse0.5")
    ksizes = df["ksize"].unique()
    ksizes.sort()

    plt.plot(ksizes, df[df["distribution"] == "uniform"]["eigvec_angle"], label="uniform")
    plt.plot(ksizes, df[df["distribution"] == "gaussian"]["eigvec_angle"], label="gaussian")
    plt.plot(ksizes, df[df["distribution"] == "dirac"]["eigvec_angle"], label="dirac")
    plt.plot(ksizes, df[df["distribution"] == "orthogonal"]["eigvec_angle"], label="orthogonal")
    plt.plot(ksizes, df[df["distribution"] == "sparse0.5"]["eigvec_angle"], label="sparse (s=0.5)")

    plt.xlabel("Kernel Size")
    plt.ylabel(r"$\texttt{eigvec}_{\mathrm{angle}}$ $[\deg]$")
    plt.title("Angle between eigenvectors for different distributions")
    plt.legend()
    plt.grid()

    plt.show()
    # plt.savefig("results/other/eigvec_angles_different_distributions.png", dpi=300)


if __name__ == "__main__":
    plot_eigvec_angles_different_distributions()
