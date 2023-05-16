from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_results(size: str) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex='all', gridspec_kw={'hspace': 0.1})

    losses = pd.read_csv(f"{size}-losses.csv")

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

    plt.savefig(f"{size}-plot.png", dpi=300)
    # plt.show()


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


if __name__ == "__main__":
    sizes = ["3x3", "6x6", "9x9", "12x12", "15x15", "18x18"]
    # plot_all_losses_from_same_startpoint("3x3", sizes)
    # plot_all_accuracies_from_same_startpoint("3x3", sizes)
    # plot_all_losses(sizes)
    plot_all_accuracies(sizes)
