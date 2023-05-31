from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def get_weight_decay(name: str) -> float:
    return float(name.split("wd")[1].split("_")[0])


def plot_pcd_results() -> None:
    fig, axs = plt.subplots(10, 2, figsize=(10, 17))
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


if __name__ == "__main__":
    plot_pcd_results()
    plt.savefig("pcd-results.png", dpi=300)
