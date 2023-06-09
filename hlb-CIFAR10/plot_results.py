from __future__ import annotations

import itertools
import json
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np
import colorsys


def plot_results(size: str) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex='all', gridspec_kw={'hspace': 0.1})

    losses = pd.read_csv(f"feature_size_experiments/{size}-losses.csv")

    ax[0].plot(losses["a-b-rebasin"].values, label="a-b-rebasin", color="red")
    ax[0].plot(losses["a-b-orig"].values, label="a-b-orig", color="blue")
    ax[0].plot(losses["b-orig-b-rebasin"].values, label="b-orig-b-rebasin", color="green")

    scale_max = max(
        np.max(losses["a-b-rebasin"].values).item(),
        np.max(losses["a-b-orig"].values).item(),
        np.max(losses["b-orig-b-rebasin"].values).item(),
    ) * 1.1
    ax[0].set_ylim(0.0, scale_max)
    ax[0].set_ylabel("Loss")

    accs = pd.read_csv(f"{size}-accuracies.csv")

    ax[1].plot(accs["a-b-rebasin"].values, label="a-b-rebasin", color="red")
    ax[1].plot(accs["a-b-orig"].values, label="a-b-orig", color="blue")
    ax[1].plot(accs["b-orig-b-rebasin"].values, label="b-orig-b-rebasin", color="green")

    scale_max = max(
        np.max(accs["a-b-rebasin"].values).item(),
        np.max(accs["a-b-orig"].values).item(),
        np.max(accs["b-orig-b-rebasin"].values).item(),
    ) * 1.1
    ax[1].set_ylim(0.0, scale_max)
    ax[1].set_ylabel("Accuracy")
    ax[1].set_xlabel("Interpolation %")

    ax[0].grid()
    ax[1].grid()
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False)
    ax[0].set_xticklabels([])
    ax[0].set_title(f"hlb-CIFAR10: filter size {size}")

    ax[1].legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        fontsize=10
    )

    plt.savefig(f"feature_size_experiments/{size}-plot.png", dpi=300)
    # plt.show()


def plot_merged_results(sizes: list[str], normalize: bool = False, show: bool = False) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharex='all', gridspec_kw={'hspace': 0.1})
    fig.subplots_adjust(top=0.9, bottom=0.2)

    title = "hlb-CIFAR10: Merged results"
    title += " (normalized starting-point)" if normalize else ""
    fig.suptitle(title, fontsize=16)

    # Get a subset of distinct dark colors from the CSS4_COLORS dictionary
    dark_colors = [
        color for color in mcolors.CSS4_COLORS
        if colorsys.rgb_to_hsv(*mcolors.to_rgb(mcolors.CSS4_COLORS[color]))[2] < 0.9
    ]
    colors = dark_colors[:len(sizes)]

    min_scale_loss = float("inf")
    max_scale_loss = -float("inf")
    min_scale_acc = float("inf")
    max_scale_acc = -float("inf")
    startpoint_loss = pd.read_csv(
        f"feature_size_experiments/{sizes[0]}-losses.csv"
    )["a-b-rebasin"].values[0].item()
    startpoint_acc = pd.read_csv(
        f"feature_size_experiments/{sizes[0]}-accuracies.csv"
    )["a-b-rebasin"].values[0].item()
    for i, size in enumerate(sizes):
        file = f"feature_size_experiments/{size}-losses.csv"
        losses = pd.read_csv(file)
        values = losses["a-b-rebasin"].values
        if normalize:
            values = values + (startpoint_loss - values[0])
        axs[0].plot(values, label=f"filter-size: {size}", color=colors[i])
        max_scale_loss = max(max_scale_loss, np.max(values).item())
        min_scale_loss = min(min_scale_loss, np.min(values).item())

        file = f"feature_size_experiments/{size}-accuracies.csv"
        accs = pd.read_csv(file)
        values = accs["a-b-rebasin"].values
        if normalize:
            values = values + (startpoint_acc - values[0])
        axs[1].plot(values, color=colors[i])
        max_scale_acc = max(max_scale_acc, np.max(values).item())
        min_scale_acc = min(min_scale_acc, np.min(values).item())

    min_scale_loss -= 0.1 * (max_scale_loss - min_scale_loss)
    max_scale_loss += 0.1 * (max_scale_loss - min_scale_loss)
    axs[0].set_ylim(min_scale_loss, max_scale_loss)
    axs[0].set_ylabel("Loss")

    min_scale_acc -= 0.1 * (max_scale_acc - min_scale_acc)
    max_scale_acc += 0.1 * (max_scale_acc - min_scale_acc)
    axs[1].set_ylim(min_scale_acc, max_scale_acc)
    axs[1].set_ylabel("Accuracy")

    axs[0].set_xlabel("Interpolation %")
    axs[1].set_xlabel("Interpolation %")

    axs[0].grid()
    axs[1].grid()

    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.1),
        ncol=3,
        fontsize=10,
        labels=[f"filter-size: {size}" for size in sizes],
        labelcolor=colors
    )

    if show:
        plt.show()
    else:
        plt.savefig(
            f"feature_size_experiments/merged-{'normalized-' if normalize else ''}plot.png",
            dpi=300
        )


def plot_all_losses(sizes: list[str]) -> None:
    plt.title("Losses: a-b-rebasin")
    plt.xlabel("Interpolation %")
    plt.ylabel("Loss")
    plt.grid()

    min_scale = 1e12
    max_scale = 0.0
    for size in sizes:
        file = f"{size}-losses.csv"
        losses = pd.read_csv(file)
        plt.plot(losses["a-b-rebasin"].values, label=f"filter-size: {size}")
        max_scale = max(max_scale, np.max(losses["a-b-rebasin"].values).item())
        min_scale = min(min_scale, np.min(losses["a-b-rebasin"].values).item())

    min_scale -= 0.1 * (max_scale - min_scale)
    max_scale += 0.1 * (max_scale - min_scale)
    plt.ylim(min_scale, max_scale)
    plt.legend()
    plt.savefig("losses-all.png", dpi=300)
    # plt.show()


def plot_all_accuracies(sizes: list[str]) -> None:
    plt.title("Accuracies: a-b-rebasin")
    plt.xlabel("Interpolation %")
    plt.ylabel("Accuracy")
    plt.grid()

    min_scale = 1e12
    max_scale = 0.0
    for size in sizes:
        file = f"{size}-accuracies.csv"
        accs = pd.read_csv(file)
        plt.plot(accs["a-b-rebasin"].values, label=f"filter-size: {size}")
        max_scale = max(max_scale, np.max(accs["a-b-rebasin"].values).item())
        min_scale = min(min_scale, np.min(accs["a-b-rebasin"].values).item())

    min_scale -= 0.1 * (max_scale - min_scale)
    max_scale += 0.1 * (max_scale - min_scale)
    plt.ylim(min_scale, max_scale)
    plt.legend()
    plt.savefig("accuracies-all.png", dpi=300)
    # plt.show()


def plot_all_losses_from_same_startpoint(cmp_size: str, sizes: list[str]) -> None:
    plt.title("Losses (normalized startpoint): a-b-rebasin")
    plt.xlabel("Interpolation %")
    plt.ylabel("Loss")
    plt.grid()

    min_scale = 1e12
    max_scale = 0.0

    startpoint = pd.read_csv(f"{cmp_size}-losses.csv")["a-b-rebasin"].values[0].item()

    for size in sizes:
        file = f"{size}-losses.csv"
        losses = pd.read_csv(file)
        values = losses["a-b-rebasin"].values
        values = values + (startpoint - values[0])
        plt.plot(values, label=f"filter-size: {size}")
        max_scale = max(max_scale, np.max(values).item())
        min_scale = min(min_scale, np.min(values).item())

    min_scale -= 0.1 * (max_scale - min_scale)
    max_scale += 0.1 * (max_scale - min_scale)
    plt.ylim(min_scale, max_scale)
    plt.legend()
    plt.savefig("losses-all-normalized-startpoint.png", dpi=300)
    # plt.show()


def plot_all_accuracies_from_same_startpoint(cmp_size: str, sizes: list[str]) -> None:
    plt.title("Accuracies (normalized startpoint): a-b-rebasin")
    plt.xlabel("Interpolation %")
    plt.ylabel("Accuracy")
    plt.grid()

    min_scale = 1e12
    max_scale = 0.0

    startpoint = pd.read_csv(f"{cmp_size}-accuracies.csv")["a-b-rebasin"].values[0].item()

    for size in sizes:
        file = f"{size}-accuracies.csv"
        accs = pd.read_csv(file)
        values = accs["a-b-rebasin"].values
        values = values + (startpoint - values[0])
        plt.plot(values, label=f"filter-size: {size}")
        max_scale = max(max_scale, np.max(values).item())
        min_scale = min(min_scale, np.min(values).item())

    min_scale -= 0.1 * (max_scale - min_scale)
    max_scale += 0.1 * (max_scale - min_scale)
    plt.ylim(min_scale, max_scale)
    plt.legend()
    plt.savefig("accuracies-all-normalized-startpoint.png", dpi=300)
    # plt.show()


def plot_merge_many_losses() -> None:
    # Manually taken from files
    losses = [1.01, 1.00, 1.02]
    losses_merge_3 = [1.98, 1.86, 1.93]
    losses_merge_6 = [2.24, 2.26, 2.26]
    losses_merge_9 = [2.28, 2.29, 2.28]
    losses_merge_12 = [2.29, 2.30, 2.29]

    gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
    fig = plt.figure(figsize=(5, 6))
    fig.subplots_adjust(hspace=0.1)
    ax = fig.add_subplot(gs[0])

    ax.set_title("Losses: MergeMany")
    ax.set_xlabel("Filter-size")
    ax.set_xticks([0, 1, 2], ["3x3", "6x6", "9x9"])
    ax.set_ylabel("Loss")
    ax.grid()
    ax.plot(losses, label="normal model (average)")
    ax.plot(losses_merge_3, label="merge 3 models")
    ax.plot(losses_merge_6, label="merge 6 models")
    ax.plot(losses_merge_9, label="merge 9 models")
    ax.plot(losses_merge_12, label="merge 12 models")

    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.15),
        ncols=2,
        fontsize=10
    )

    plt.savefig("merge_many/losses.png", dpi=300)


def plot_merge_many_accuracies() -> None:
    # Manually taken from files in merge_many folder
    accuracies = [0.94, 0.93, 0.92]
    accuracies_merge_3 = [0.82, 0.86, 0.83]
    accuracies_merge_6 = [0.44, 0.38, 0.43]
    accuracies_merge_9 = [0.24, 0.17, 0.27]
    accuracies_merge_12 = [0.21, 0.13, 0.19]

    gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
    fig = plt.figure(figsize=(5, 6))
    fig.subplots_adjust(hspace=0.1)
    ax = fig.add_subplot(gs[0])

    ax.set_title("Accuracies MergeMany")
    ax.set_xlabel("Filter-size")
    ax.set_xticks([0, 1, 2], ["3x3", "6x6", "9x9"])
    ax.set_ylabel("Accuracy")
    ax.grid()
    ax.plot(accuracies_merge_3, label="merge 3 models")
    ax.plot(accuracies_merge_6, label="merge 6 models")
    ax.plot(accuracies_merge_9, label="merge 9 models")
    ax.plot(accuracies_merge_12, label="merge 12 models")
    ax.plot(accuracies, label="normal model (average)")

    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.15),
        ncols=2,
        fontsize=10
    )

    plt.savefig("merge_many/accuracies.png", dpi=300)


def plot_merge_many_different_datasets(show: bool = False) -> None:
    with open("merge_many/train_different_datasets/losses.txt", "r") as f:
        losses: dict[str, dict[int, dict[str, list[float]]]] = json.loads(f.read())
    with open("merge_many/train_different_datasets/accuracies.txt", "r") as f:
        accuracies: dict[str, dict[int, dict[str, list[float]]]] = json.loads(f.read())

    plot_group__merge_many_different_datasets(
        losses, accuracies, "models", normalize=False, show=show
    )
    plot_group__merge_many_different_datasets(
        losses, accuracies, "epochs", normalize=False, show=show
    )
    plot_group__merge_many_different_datasets(
        losses, accuracies, "models", normalize=True, show=show
    )
    plot_group__merge_many_different_datasets(
        losses, accuracies, "epochs", normalize=True, show=show
    )


def plot_group__merge_many_different_datasets(
        losses: dict[str, dict[int, dict[str, list[float]]]],
        accuracies: dict[str, dict[int, dict[str, list[float]]]],
        key: str,
        normalize: bool,
        show: bool = False
) -> None:
    for (quantity, loss_data), (quantity_control, acc_data) in zip(
            losses[key].items(), accuracies[key].items()
    ):
        assert quantity == quantity_control
        assert len(loss_data['avg_model']) == len(acc_data['avg_model'])

        if normalize:
            loss_data = normalize_data(loss_data)
            acc_data = normalize_data(acc_data)

        gs = gridspec.GridSpec(2, 2, height_ratios=[9, 1])
        fig = plt.figure(figsize=(9, 5))
        fig.subplots_adjust(hspace=0.1, wspace=0.3, left=0.1, right=0.95)
        axs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        if normalize:
            fig.suptitle(f"Results (normalized): MergeMany ({key}: {quantity})")
        else:
            fig.suptitle(f"Results: MergeMany ({key}: {quantity})")

        ticks = np.arange(len(loss_data['avg_model']))
        labels = (ticks + 1) * (3 if key == 'epochs' else 10)
        axs[0].set_xticks(ticks, labels)
        axs[1].set_xticks(ticks, labels)

        axs[0].set_title("Losses")
        axs[1].set_title("Accuracies")

        axs[0].set_xlabel(f"# of {'epochs' if key == 'models' else 'models'}")
        axs[1].set_xlabel(f"# of {'epochs' if key == 'models' else 'models'}")

        axs[0].set_ylabel("Loss")
        axs[1].set_ylabel("Accuracy")

        axs[0].grid()
        axs[1].grid()

        for model_name, values in loss_data.items():
            axs[0].plot(values, label=model_name)
        for model_name, values in acc_data.items():
            axs[1].plot(values)

        fig.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.13),
            ncols=4,
            fontsize=10
        )
        if show:
            plt.show()
        else:
            name = f"{quantity}{key}"
            name = "results-normalized-" + name if normalize else "results-" + name
            name = f"merge_many/train_different_datasets/{name}.png"
            plt.savefig(name, dpi=300)
        plt.clf()
        plt.cla()
        plt.close()


def plot_pcd_weight_decay(show: bool = False) -> None:
    files = os.listdir("weight_decay_experiments")
    files = [file for file in files if file.endswith(".csv")]
    losses = []
    accuracies = []
    for file in files:
        if "loss" in file:
            losses.append(file)
        elif "acc" in file:
            accuracies.append(file)

    losses.sort(key=lambda name: float(name.split("wd")[1].split(".csv")[0]))
    accuracies.sort(key=lambda name: float(name.split("wd")[1].split(".csv")[0]))

    fig, axs = plt.subplots(len(losses), 2, figsize=(10, 17))
    fig.subplots_adjust(top=0.94, bottom=0.06)

    fig.suptitle("PermutationCoordinateDescent at different weight_decay values")

    for i, (loss_file, acc_file) in enumerate(zip(losses, accuracies)):
        loss_data = pd.read_csv(f"weight_decay_experiments/{loss_file}")
        acc_data = pd.read_csv(f"weight_decay_experiments/{acc_file}")

        axs[i][0].set_title(f"weight_decay: {loss_file.split('wd')[1].split('.csv')[0]}")
        axs[i][1].set_title(f"weight_decay: {acc_file.split('wd')[1].split('.csv')[0]}")

        if i == len(losses) - 1:
            axs[i][0].set_xlabel("Interpolation %")
            axs[i][1].set_xlabel("Interpolation %")
        else:
            axs[i][0].set_xticklabels([])
            axs[i][1].set_xticklabels([])

        axs[i][0].set_ylabel("Loss")
        axs[i][1].set_ylabel("Accuracy")

        axs[i][0].grid()
        axs[i][1].grid()

        axs[i][0].plot(loss_data["a-b-rebasin"], color="red")
        axs[i][0].plot(loss_data["a-b-orig"], color="green")
        axs[i][0].plot(loss_data["b-orig-b-rebasin"], color="blue")

        axs[i][1].plot(acc_data["a-b-rebasin"], color="red")
        axs[i][1].plot(acc_data["a-b-orig"], color="green")
        axs[i][1].plot(acc_data["b-orig-b-rebasin"], color="blue")

    fig.legend(
        labels=["a-b-rebasin", "a-b-orig", "b-orig-b-rebasin"],
        loc="lower center",
        ncol=3,
        labelcolor=["red", "green", "blue"],
    )

    if show:
        plt.show()
    else:
        plt.savefig("weight_decay_experiments/weight_decay.png", dpi=300)


def plot_pcd_feature_size(show: bool = False) -> None:
    root = "feature_size_experiments"
    files = os.listdir(root)
    files = [file for file in files if file.endswith(".csv")]
    losses = []
    accuracies = []
    for file in files:
        if "loss" in file:
            losses.append(file)
        elif "acc" in file:
            accuracies.append(file)

    losses.sort(key=lambda name: int(name.split("x")[0]))
    accuracies.sort(key=lambda name: int(name.split("x")[0]))

    fig, axs = plt.subplots(len(losses), 2, figsize=(10, 12))
    fig.subplots_adjust(top=0.93, bottom=0.08)

    fig.suptitle("PermutationCoordinateDescent at different feature sizes")

    for i, (loss_file, acc_file) in enumerate(zip(losses, accuracies)):
        loss_data = pd.read_csv(f"{root}/{loss_file}")
        acc_data = pd.read_csv(f"{root}/{acc_file}")

        fs = loss_file.split("x")[0]

        axs[i][0].set_title(f"feature_size: {fs}x{fs}")
        axs[i][1].set_title(f"feature_size: {fs}x{fs}")

        if i == len(losses) - 1:
            axs[i][0].set_xlabel("Interpolation %")
            axs[i][1].set_xlabel("Interpolation %")
        else:
            axs[i][0].set_xticklabels([])
            axs[i][1].set_xticklabels([])

        axs[i][0].set_ylabel("Loss")
        axs[i][1].set_ylabel("Accuracy")

        axs[i][0].grid()
        axs[i][1].grid()

        axs[i][0].plot(loss_data["a-b-rebasin"], color="red")
        axs[i][0].plot(loss_data["a-b-orig"], color="green")
        axs[i][0].plot(loss_data["b-orig-b-rebasin"], color="blue")

        axs[i][1].plot(acc_data["a-b-rebasin"], color="red")
        axs[i][1].plot(acc_data["a-b-orig"], color="green")
        axs[i][1].plot(acc_data["b-orig-b-rebasin"], color="blue")

    fig.legend(
        labels=["a-b-rebasin", "a-b-orig", "b-orig-b-rebasin"],
        loc="lower center",
        ncol=3,
        labelcolor=["red", "green", "blue"],
    )

    if show:
        plt.show()
    else:
        plt.savefig(f"{root}/results.png", dpi=300)


def normalize_data(data: dict[str, list[float]], norm_key: str = 'merged_model') -> dict[str, list[float]]:
    new_data: dict[str, list[float]] = {}
    for model_name, values in data.items():
        new_data[model_name] = [value / data[norm_key][i] for i, value in enumerate(values)]
    return new_data


def plot_loss_predictiveness(wd: float, ks: int) -> None:
    df = pd.read_csv(f"loss_predictiveness/loss_predictiveness_before_bn_recalc_wd{wd}_ks{ks}.csv")

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    fig.subplots_adjust(top=0.85, bottom=0.2, wspace=0.25, left=0.1, right=0.98)
    fig.suptitle(f"weight_decay: {wd}, kernel_size: {ks}")

    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")

    axs[0].set_xlabel("Interpolation %")
    axs[1].set_xlabel("Interpolation %")

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    axs[0].grid()
    axs[1].grid()

    axs[0].plot(df["step"], df["loss_bn_a"], label="BN model A")
    axs[0].plot(df["step"], df["loss_bn_b"], label="BN model B", linestyle="-.")
    axs[0].plot(df["step"], df["loss_reset"], label="BN reset")
    axs[0].plot(df["step"], df["loss_recalc"], label="BN recalculated")

    axs[1].plot(df["step"], df["acc_bn_a"])
    axs[1].plot(df["step"], df["acc_bn_b"], linestyle="-.")
    axs[1].plot(df["step"], df["acc_reset"])
    axs[1].plot(df["step"], df["acc_recalc"])
    fig.legend(
        loc="lower center",
        ncol=4,
        # bbox_to_anchor=(0.5, 0.05),
    )

    # plt.show()
    plt.savefig(f"loss_predictiveness/loss_predictiveness_before_bn_recalc_wd{wd}_ks{ks}.png", dpi=300)


if __name__ == "__main__":
    wds = (0.0, 0.001, 0.05, 0.075, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8, 1.0)
    kss = (3, 4)
    for wd, ks in itertools.product(wds, kss):
        plot_loss_predictiveness(wd, ks)
