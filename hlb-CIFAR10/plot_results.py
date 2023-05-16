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


if __name__ == "__main__":
    for size in ["3x3", "6x6", "9x9", "12x12", "15x15"]:
        plot_results(size)

