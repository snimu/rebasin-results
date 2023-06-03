from __future__ import annotations

import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_count_permutations() -> None:
    results = pd.read_csv("results/merge-many/count_permutations_wd0.0-0.2_hf400.csv")
    plt.title("MergeMany: num_models=3, hidden_features=400")
    plt.plot(
        results["weight_decay"],
        results["num_permutations"],
    )
    plt.grid()
    plt.ylabel("Number of Permutations")
    plt.xlabel("Weight Decay")
    plt.show()
    # plt.savefig("results/merge-many/count_permutations_wd0.0-0.2_hf400.png", dpi=300)


def normalize(data: pd.DataFrame, key_loss: str, key_acc: str) -> pd.DataFrame:
    ndata = copy.deepcopy(data)
    norm_factors_loss = ndata[key_loss].values
    norm_factors_acc = ndata[key_acc].values
    ndata["loss_avg"] = ndata["loss_avg"] / norm_factors_loss
    ndata["loss_merged"] = ndata["loss_merged"] / norm_factors_loss
    ndata["acc_avg"] = ndata["acc_avg"] / norm_factors_acc
    ndata["acc_merged"] = ndata["acc_merged"] / norm_factors_acc
    return ndata


if __name__ == "__main__":
    plot_count_permutations()
