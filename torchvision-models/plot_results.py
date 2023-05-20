from __future__ import annotations

import os

import matplotlib.pyplot as plt  # type: ignore[import]
import pandas as pd


def plot_results(model: str, metric: str) -> None:
    results = pd.read_csv(os.path.join(model, f"{model}_{metric}.csv"))

    plt.plot(results.a_b_original, label="a_b_original")
    plt.plot(results.a_b_rebasin, label="a_b_rebasin")
    plt.plot(results.b_original_b_rebasin, label="b_original_b_rebasin")

    scale_max = max(
        results.a_b_original.values.max().item(),
        results.a_b_rebasin.values.max().item(),
        results.b_original_b_rebasin.values.max().item(),
    )
    scale_min = min(
        results.a_b_original.values.min().item(),
        results.a_b_rebasin.values.min().item(),
        results.b_original_b_rebasin.values.min().item(),
    )
    height = scale_max - scale_min

    plt.ylim(scale_min - 0.1 * height, scale_max + 0.1 * height)
    metric_name = "Loss" if metric == "losses" else f"Accuracy (Top {metric[-1]})"
    plt.ylabel(metric_name)
    plt.xlabel("Interpolation step")
    plt.legend()
    plt.grid()

    plt.title(f"{model}: {metric_name}")
    plt.savefig(os.path.join(model, f"{model}_{metric}.png"))
    # plt.show()


if __name__ == "__main__":
    plot_results("mobilenet_v3_large", "accuracies5")
