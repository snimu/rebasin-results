from __future__ import annotations

import os
import json
from typing import Any


def get_appropriate_lines(text: str) -> tuple[str, str, str, str]:
    blocks = text.split("\n\n")[1:]
    avg_model_line, merged_model_line = blocks[0].split("\n")
    control_model_line_1 = blocks[1].split("\n")[0]
    control_model_line_n = blocks[1].split("\n")[-1]

    return avg_model_line, merged_model_line, control_model_line_1, control_model_line_n


def extract_loss_and_accuracy(line: str) -> tuple[float, float]:
    line = line.strip()
    loss = float(line.split("Loss: ")[1].split(",")[0])
    accuracy = float(line.split("Acc: ")[1])
    return loss, accuracy


def add_info(
        losses: dict[str, Any],
        accuracies: dict[str, Any],
        text: str,
        metric: str,
        count: int
) -> None:
    lines = get_appropriate_lines(text)
    avg_model_loss, avg_model_acc = extract_loss_and_accuracy(lines[0])
    merged_model_loss, merged_model_acc = extract_loss_and_accuracy(lines[1])
    control_model_loss_1, control_model_acc_1 = extract_loss_and_accuracy(lines[2])
    control_model_loss_n, control_model_acc_n = extract_loss_and_accuracy(lines[3])

    if not losses[metric] or count not in losses[metric].keys():
        losses[metric][count] = {
            "avg_model": [],
            "merged_model": [],
            "control_model_1": [],
            "control_model_n": []
        }
    losses[metric][count]["avg_model"].append(avg_model_loss)
    losses[metric][count]["merged_model"].append(merged_model_loss)
    losses[metric][count]["control_model_1"].append(control_model_loss_1)
    losses[metric][count]["control_model_n"].append(control_model_loss_n)

    if not accuracies[metric] or count not in accuracies[metric].keys():
        accuracies[metric][count] = {
            "avg_model": [],
            "merged_model": [],
            "control_model_1": [],
            "control_model_n": []
        }
    accuracies[metric][count]["avg_model"].append(avg_model_acc)
    accuracies[metric][count]["merged_model"].append(merged_model_acc)
    accuracies[metric][count]["control_model_1"].append(control_model_acc_1)
    accuracies[metric][count]["control_model_n"].append(control_model_acc_n)


def get_losses_and_accuracies() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Get the losses and accuracies from the files in the current directory.

    :return: Losses and accuracies per number of epochs and per number of models.
    """
    losses = {"epochs": {}, "models": {}}
    accuracies = {"epochs": {}, "models": {}}

    files = os.listdir(".")
    files = [file for file in files if file.endswith(".txt") and not file.startswith(".")]
    files.sort(key=lambda x: int(x.split("_")[4][:-6]))  # Sorted by model number
    for file in files:
        with open(file, "r") as f:
            text = f.read()

        num_epochs = int(file.split("_")[5][:-6])
        add_info(losses, accuracies, text, "epochs", num_epochs)

    files.sort(key=lambda x: int(x.split("_")[5][:-6]))  # Sorted by number of epochs
    for file in files:
        with open(file, "r") as f:
            text = f.read()

        num_models = int(file.split("_")[4][:-6])
        add_info(losses, accuracies, text, "models", num_models)

    return losses, accuracies


def main() -> None:
    losses, accuracies = get_losses_and_accuracies()
    with open("accuracies.txt", "w") as f:
        f.write(json.dumps(accuracies))
    with open("losses.txt", "w") as f:
        f.write(json.dumps(losses))


if __name__ == "__main__":
    main()
