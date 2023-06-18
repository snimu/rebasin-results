"""Train an MLP on MNIST."""
from __future__ import annotations

import argparse
import copy
import os
import itertools
import shutil
from collections import Counter
from typing import Any, Sequence

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch
import torchvision
import rebasin
import pandas as pd
from torchview import draw_graph
from tqdm import tqdm


class MLP(nn.Module):
    def __init__(
            self,
            in_features: int = 28*28,
            out_features: int = 10,
            hidden_features: int | None = None,
            num_layers: int = 5,
            use_layer_norm: bool = True
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(
                f"num_layers must be positive, got {num_layers}"
            )
        if num_layers == 1:
            self.model = nn.Linear(in_features, out_features)
            return

        hidden_features = hidden_features or 20

        self.sequential = nn.Sequential()

        self.sequential.append(nn.Linear(in_features, hidden_features))
        self.sequential.append(nn.ReLU())
        if use_layer_norm:
            self.sequential.append(nn.LayerNorm(hidden_features))

        for _ in range(num_layers - 2):
            self.sequential.append(nn.Linear(hidden_features, hidden_features))
            self.sequential.append(nn.ReLU())
            if use_layer_norm:
                self.sequential.append(nn.LayerNorm(hidden_features))

        self.sequential.append(nn.Linear(hidden_features, out_features))
        self.sequential.append(nn.Sigmoid())

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.sequential(input_tensor)


def train_mnist(
        in_features: int = 28 * 28,
        out_features: int = 10,
        hidden_features: int | None = None,
        num_layers: int = 5,
        use_layer_norm: bool = True,
        epochs: int = 2,
        learning_rate: float = 7e-5,
        weight_decay: float = 0.0,
        loop: tqdm[Any] | None = None,
        verbose: bool = False,
) -> MLP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLP(
        in_features, out_features, hidden_features, num_layers, use_layer_norm
    ).to(device)
    mlp.train()

    train_loader = DataLoader(
        MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=32,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        mlp.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = mlp(inputs.view(inputs.shape[0], -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0 and verbose:
                feedback = f"Epoch {epoch}, batch {i}: loss {loss.item()}"
                if loop is not None:
                    loop.write(feedback)
                else:
                    print(feedback)

    return mlp


def eval_fn(
        model: MLP,
        device: torch.device
) -> tuple[float, float]:
    dataloader = DataLoader(
        MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=32,
        shuffle=False,
    )
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.view(inputs.shape[0], -1))
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            labels: torch.Tensor
            predicted: torch.Tensor
            correct += (predicted == labels).sum().item()

    model.train()
    return loss / len(dataloader), correct / total


def get_filenames(directory):
    filenames = os.listdir(directory)
    if "model_0.pt" in filenames:
        filenames.remove("model_0.pt")
    if "model_1.pt" in filenames:
        filenames.remove("model_1.pt")

    filenames.sort(key=lambda x: float(x[-8:-3]))
    return filenames


def test_permutation_coordinate_descent(
        learning_rate: float = 7e-5,
        weight_decay: float = 0.0,
        epochs: int = 1,
        hidden_features: int | None = None,
        verbose: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a = train_mnist(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        hidden_features=hidden_features,
    ).to(device)
    model_b = train_mnist(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        hidden_features=hidden_features,
    ).to(device)
    model_b_orig = copy.deepcopy(model_b).to(device)
    x = torch.randn(32, 28*28).to(device)
    pcd = rebasin.PermutationCoordinateDescent(
         model_a, model_b, x, device_b=device
    )
    pcd.rebasin()

    # Check that model_b has changed
    diff_sum = 0.0
    model_b_sum = 0.0
    for param_b, param_b_orig in zip(model_b.parameters(), model_b_orig.parameters()):
        diff_sum += torch.sum(torch.abs(param_b - param_b_orig))
        model_b_sum += torch.sum(torch.abs(param_b_orig))

    assert diff_sum / model_b_sum > 0.02

    # Check that output stays the same
    assert torch.allclose(model_b(x), model_b_orig(x))

    # Interpolate A B-rebasin
    directory = "models-a-b-rebasin"
    os.makedirs(directory, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_a, model_b],
        devices=[device, device],
        device_interp=device,
        savedir=directory,
        logging_level="info" if verbose else "warning"
    )
    interp.interpolate(steps=99)

    # Interpolate A B-original
    directory = "models-a-b-original"
    os.makedirs(directory, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_a, model_b_orig],
        devices=[device, device],
        device_interp=device,
        savedir=directory,
        logging_level="info" if verbose else "warning"
    )
    interp.interpolate(steps=99)

    # Interpolate B-original B-rebasin
    directory = "models-b-original-b-rebasin"
    os.makedirs(directory, exist_ok=True)
    interp = rebasin.interpolation.LerpSimple(
        models=[model_b_orig, model_b],
        devices=[device, device],
        device_interp=device,
        savedir=directory,
        logging_level="info" if verbose else "warning"
    )
    interp.interpolate(steps=99)

    # EVALUATION

    # Original models
    loss_a, acc_a = eval_fn(model_a, device)
    loss_b, acc_b = eval_fn(model_b, device)
    loss_b_orig, acc_b_orig = eval_fn(model_b_orig, device)

    # Interpolated models
    working_model = MLP(28*28, 10, hidden_features)

    directory = "models-a-b-rebasin"
    files = get_filenames(directory)
    loss_interp_a_b_rebasin = []
    acc_interp_a_b_rebasin = []
    print(f"Evaluating {directory}")
    for file in tqdm(files):
        working_model.load_state_dict(torch.load(os.path.join(directory, file)))
        working_model.to(device)
        loss, acc = eval_fn(working_model, device)
        loss_interp_a_b_rebasin.append(loss)
        acc_interp_a_b_rebasin.append(acc)

    directory = "models-a-b-original"
    files = get_filenames(directory)
    loss_interp_a_b_original = []
    acc_interp_a_b_original = []
    print(f"Evaluating {directory}")
    for file in tqdm(files):
        working_model.load_state_dict(torch.load(os.path.join(directory, file)))
        working_model.to(device)
        loss, acc = eval_fn(working_model, device)
        loss_interp_a_b_original.append(loss)
        acc_interp_a_b_original.append(acc)

    directory = "models-b-original-b-rebasin"
    files = get_filenames(directory)
    loss_interp_b_original_b_rebasin = []
    acc_interp_b_original_b_rebasin = []
    print(f"Evaluating {directory}")
    for file in tqdm(files):
        working_model.load_state_dict(torch.load(os.path.join(directory, file)))
        working_model.to(device)
        loss, acc = eval_fn(working_model, device)
        loss_interp_b_original_b_rebasin.append(loss)
        acc_interp_b_original_b_rebasin.append(acc)

    losses = {
        "a-b-orig": [loss_a, *loss_interp_a_b_original, loss_b_orig],
        "a-b-rebasin": [loss_a, *loss_interp_a_b_rebasin, loss_b],
        "b-orig-b-rebasin": [loss_b_orig, *loss_interp_b_original_b_rebasin, loss_b],
    }
    accs = {
        "a-b-orig": [acc_a, *acc_interp_a_b_original, acc_b_orig],
        "a-b-rebasin": [acc_a, *acc_interp_a_b_rebasin, acc_b],
        "b-orig-b-rebasin": [acc_b_orig, *acc_interp_b_original_b_rebasin, acc_b],
    }

    # Save results
    df = pd.DataFrame(losses)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/losses_wd{weight_decay}_hf{hidden_features}.csv")
    df = pd.DataFrame(accs)
    df.to_csv(f"results/accuracies_wd{weight_decay}_hf{hidden_features}.csv")

    # Delete old models
    # (I'm only interested in the results, not the models themselves,
    # and I need to free the directories for the next run)
    shutil.rmtree("models-a-b-rebasin")
    shutil.rmtree("models-a-b-original")
    shutil.rmtree("models-b-original-b-rebasin")


def test_merge_many(
        hidden_features: Sequence[int],
        weight_decays: Sequence[float],
        num_models: Sequence[int],
        learning_rate: float = 7e-5,
        epochs: int = 1,
        verbose: bool = True,
) -> None:
    assert isinstance(hidden_features, Sequence)
    assert isinstance(weight_decays, Sequence)
    assert isinstance(num_models, Sequence)
    assert all(isinstance(x, int) for x in hidden_features)
    assert all(isinstance(x, float) for x in weight_decays)
    assert all(isinstance(x, int) for x in num_models)

    results = {
        "hidden_features": [],
        "weight_decay": [],
        "num_models": [],
        "loss_avg": [],
        "acc_avg": [],
        "loss_merged": [],
        "acc_merged": [],
    }

    loop = tqdm(
        itertools.product(hidden_features, weight_decays, num_models),
        total=len(hidden_features) * len(weight_decays) * len(num_models),
        disable=not verbose,
    )
    for feature_num, weight_decay, model_num in loop:
        if verbose:
            loop.set_description(f"hf={feature_num}, wd={weight_decay}, nm={model_num}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            loop.write(f"hf={feature_num}, wd={weight_decay}, nm={model_num}")
        models = []
        for i in range(model_num):
            models.append(
                train_mnist(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    hidden_features=feature_num,
                ).to(device)
            )

        x = torch.randn(64, 28*28).to(device)

        losses = []
        accs = []
        for model in models:
            loss, acc = eval_fn(model, device)
            losses.append(loss)
            accs.append(acc)
        loss_avg = sum(losses) / len(losses)
        acc_avg = sum(accs) / len(accs)

        if verbose:
            loop.write(f"Average loss: {loss_avg}")
            loop.write(f"Average accuracy: {acc_avg}")
        mm = rebasin.MergeMany(
            models=models,
            working_model=MLP(28 * 28, 10, feature_num),
            input_data=x,
            device=device,
        )
        merged_model: MLP = mm.run()  # type: ignore
        loss_merged, acc_merged = eval_fn(merged_model, device)

        if verbose:
            loop.write(f"Merged model loss: {loss_merged}")
            loop.write(f"Merged model accuracy: {acc_merged}\n")

        results["hidden_features"].append(feature_num)
        results["weight_decay"].append(weight_decay)
        results["num_models"].append(model_num)
        results["loss_avg"].append(loss_avg)
        results["acc_avg"].append(acc_avg)
        results["loss_merged"].append(loss_merged)
        results["acc_merged"].append(acc_merged)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/merge_many.csv")


def print_model(hidden_features: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(64, 28*28)
    x = x.to(device)
    model = MLP(28*28, 10, hidden_features)
    model.to(device)

    graph = draw_graph(model, x, depth=1e12).visual_graph
    graph.render(f"graph_hf{hidden_features}")


def show_permutations(hidden_features) -> None:
    ma = MLP(28*28, 10, hidden_features)
    mb = MLP(28*28, 10, hidden_features)
    x = torch.randn(64, 28*28)

    pcd = rebasin.PermutationCoordinateDescent(ma, mb, x)

    with open(f"permutations_hf{hidden_features}.txt", "w") as f:
        f.write(repr(pcd.pinit.model_graph))


def tune_training_parameters() -> None:
    """Just tune learning_rate and num_layers;
    I'll do sweeps over weight-decay and hidden_features later.

    Don't want to install ray[tune], so I'll do it manually.
    Isn't very important anyway.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrs = (1e-3, 5e-4, 1e-4, 5e-5)
    num_layers = (3, 4, 5, 6, 7)

    print("lr test at fixed num_layers")
    best_loss = float("inf")
    best_acc = -float("inf")
    best_lr = 1e-3
    for lr in tqdm(lrs):
        model = train_mnist(learning_rate=lr)
        loss, acc = eval_fn(model, device)
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_lr = lr

    print(f"best lr={best_lr}, loss={best_loss}, acc={best_acc}")
    print()

    print("num_layers test at fixed lr")
    best_loss = float("inf")
    best_acc = -float("inf")
    best_num_layers = 1
    for num_layer in tqdm(num_layers):
        model = train_mnist(num_layers=num_layer, learning_rate=best_lr)
        loss, acc = eval_fn(model, device)
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_num_layers = num_layer

    print(f"best num_layers={best_num_layers}, loss={best_loss}, acc={best_acc}")
    print()

    print("lr and num_layers test")
    lrs = [
        0.1 * best_lr,
        0.4 * best_lr,
        0.7 * best_lr,
        best_lr,
        4 * best_lr,
        7 * best_lr,
        10 * best_lr
    ]
    num_layers = [
        best_num_layers - 2,
        best_num_layers - 1,
        best_num_layers,
        best_num_layers + 1,
        best_num_layers + 2
    ]
    best_loss = float("inf")
    best_acc = -float("inf")
    best_lr = 1e-3
    best_num_layers = 1
    loop = tqdm(itertools.product(lrs, num_layers))
    for lr, num_layer in loop:
        loop.set_description(f"{lr=}, {num_layer=}")
        model = train_mnist(learning_rate=lr, num_layers=num_layer, loop=loop)
        loss, acc = eval_fn(model, device)
        if loss < best_loss:
            best_loss = loss
            best_acc = acc
            best_lr = lr
            best_num_layers = num_layer

    print(
        f"best lr={best_lr}, best num_layers={best_num_layers}, "
        f"loss={best_loss}, acc={best_acc}"
    )


def full_wd_hf_sweep_merge_many() -> None:
    weight_decays = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2)
    hidden_features = (100, 400, 800, 1200, 1600, 2000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "weight_decay": [],
        "hidden_features": [],
        "loss_avg": [],
        "acc_avg": [],
        "loss_merged": [],
        "acc_merged": []
    }

    loop = tqdm(
        itertools.product(weight_decays, hidden_features),
        total=len(weight_decays) * len(hidden_features)
    )
    for wd, hf in loop:
        loop.set_description(f"{wd=}, {hf=}")
        models = [train_mnist(weight_decay=wd, hidden_features=hf).to(device) for _ in range(3)]
        loss_avg, acc_avg = 0.0, 0.0
        for model in models:
            loss, acc = eval_fn(model, device)
            loss_avg += loss
            acc_avg += acc
        loss_avg /= len(models)
        acc_avg /= len(models)

        mm = rebasin.MergeMany(
            models,
            MLP(hidden_features=hf).to(device),
            torch.randn(64, 28*28).to(device),
            device=device
        )
        mm.run()
        loss_merged, acc_merged = eval_fn(mm.merged_model, device)

        results["weight_decay"].append(wd)
        results["hidden_features"].append(hf)
        results["loss_avg"].append(loss_avg)
        results["acc_avg"].append(acc_avg)
        results["loss_merged"].append(loss_merged)
        results["acc_merged"].append(acc_merged)

        loop.write(f"{wd=}, {hf=}, {loss_avg=}, {acc_avg=}, {loss_merged=}, {acc_merged=}")

    df = pd.DataFrame(results)
    df.to_csv(
        f"full_wd{min(weight_decays)}-{max(weight_decays)}"
        f"_hf{min(hidden_features)}-{max(hidden_features)}"
        f"_sweep.csv",
        index=False
    )


def compare_output_statistics(
        feature_nums: list[int],
        weight_decays: list[float],
        model_nums: list[float],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "weight_decay": [],
        "hidden_features": [],
        "num_models": [],
        "max_avg": [],
        "std_avg": [],
        "max_merged": [],
        "std_merged": [],
        "loss_ratio": [],
        "acc_ratio": []
    }

    last_settings = {"wd": -1, "hf": -1, "nm": -1}
    models = []
    loop = tqdm(
        itertools.product(weight_decays, feature_nums, model_nums),
        total=len(weight_decays) * len(feature_nums) * len(model_nums)
    )
    for wd, hf, nm in loop:
        loop.set_description(f"{wd=}, {hf=}, {nm=}")
        if wd != last_settings["wd"] or hf != last_settings["hf"]:
            last_settings["wd"] = wd
            last_settings["hf"] = hf
            del models  # force freeing-up of memory
            models = list(
                train_mnist(hidden_features=hf, weight_decay=wd).to(device)
                for _ in range(nm)
            )
        elif nm > last_settings["nm"]:
            models.extend(
                train_mnist(hidden_features=hf, weight_decay=wd).to(device)
                for _ in range(last_settings["nm"], nm)
            )
            last_settings["nm"] = nm
        elif nm < last_settings["nm"]:
            models = models[:nm]
            last_settings["nm"] = nm
        else:
            continue

        maxs, stds = [], []
        losses, accs = [], []
        for model in models:
            m, s = output_statistics(model, device)
            maxs.append(sum(m) / len(m))
            stds.append(sum(s) / len(s))
            loss, acc = eval_fn(model, device)
            losses.append(loss)
            accs.append(acc)
        max_avg = sum(maxs) / len(maxs)
        std_avg = sum(stds) / len(stds)
        loss_avg = sum(losses) / len(losses)
        acc_avg = sum(accs) / len(accs)

        mm = rebasin.MergeMany(
            models,
            MLP(hidden_features=hf).to(device),
            torch.randn(64, 28*28).to(device),
            device=device
        )
        mm.run()

        maxs, stds = output_statistics(mm.merged_model, device)
        max_merged = sum(maxs) / len(maxs)
        std_merged = sum(stds) / len(stds)
        loss_merged, acc_merged = eval_fn(mm.merged_model, device)

        loop.write(f"{wd=}, {hf=}, {nm=}, {max_avg=}, {std_avg=}, {max_merged=}, {std_merged=}")

        results["weight_decay"].append(wd)
        results["hidden_features"].append(hf)
        results["num_models"].append(nm)
        results["max_avg"].append(max_avg)
        results["std_avg"].append(std_avg)
        results["max_merged"].append(max_merged)
        results["std_merged"].append(std_merged)
        results["loss_ratio"].append(loss_merged / loss_avg)
        results["acc_ratio"].append(acc_merged / acc_avg)

    df = pd.DataFrame(results)
    df.to_csv(
        f"compare_output_statistics"
        f"_wd{min(weight_decays)}-{max(weight_decays)}"
        f"_hf{min(feature_nums)}-{max(feature_nums)}"
        f"_nm{min(model_nums)}-{max(model_nums)}.csv",
        index=False
    )


@torch.no_grad()
def output_statistics(model: MLP, device: torch.device) -> tuple[list[float], list[float]]:
    dataloader = DataLoader(
        MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=32,
        shuffle=False,
    )
    model.eval()

    maximums = []
    stds = []
    for x, _ in dataloader:
        x = x.to(device)
        output = model(x.view(x.shape[0], -1))
        maximums.append(output.max().item())
        stds.append(output.std().item())

    return maximums, stds


def count_permutations(hidden_features: int, weight_decays: list[float]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "weight_decay": [],
        "num_permutations": []
    }

    loop = tqdm(weight_decays)
    for wd in loop:
        loop.set_description(f"{wd=}")
        models = list(
            train_mnist(hidden_features=hidden_features, weight_decay=wd).to(device)
            for _ in range(3)
        )

        # To find out the number of permutations, these models will have to be copied
        models_orig = copy.deepcopy(models)

        mm = rebasin.MergeMany(
            models,
            MLP(hidden_features=hidden_features).to(device),
            torch.randn(64, 28*28).to(device),
            device=device
        )
        mm.run()

        # Count the permutations
        num_permutations = 0
        for model, model_orig in zip(mm.models, models_orig):
            num_permutations += sum(
                p.numel() - (p == p_orig).sum().item()
                for p, p_orig in zip(model.parameters(), model_orig.parameters())
            )

        num_permutations /= len(models)
        results["weight_decay"].append(wd)
        results["num_permutations"].append(num_permutations)

    df = pd.DataFrame(results)
    df.to_csv(
        f"count_permutations"
        f"_wd{min(weight_decays)}-{max(weight_decays)}"
        f"_hf{hidden_features}.csv",
        index=False
    )


def choose_most_common(outputs: list[torch.Tensor]) -> torch.Tensor:
    """Return a tensor of zeros with a one where most outputs have their maximum."""
    argmax_counts = Counter([output.argmax(dim=1) for output in outputs])
    most_common = argmax_counts.most_common(1)[0][0]
    output = torch.zeros_like(outputs[0])
    for i, mc in enumerate(most_common):
        output[i, mc] = 1
    return output


def nfold_forward(model: MLP, x: torch.Tensor, n: int) -> torch.Tensor:
    outputs = []
    mean_x = x.mean()
    for _ in range(n):
        # Add a little noise to the input so that the model doesn't always
        # predict the same class
        noise = torch.randn_like(x) * mean_x / 100
        outputs.append(model(x + noise))
    return choose_most_common(outputs)


def eval_fn_nfold(model: MLP, device: torch.device, n: int) -> tuple[float, float]:
    dataloader = DataLoader(
        MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ),
        batch_size=32,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = nfold_forward(model, x.view(x.shape[0], -1), n)
            loss += criterion(output, y).item()
            correct += (output.argmax(1) == y).sum().item()
            total += y.numel()

    return loss / total, correct / total


def test_merge_many_nfold(
        hidden_features: int,
        weight_decay: float,
        num_models: int,
        forward_pass_nums: list[int],
) -> None:
    results = {
        "nforward": [],
        "loss_avg": [],
        "acc_avg": [],
        "loss_merged": [],
        "acc_merged": [],
    }
    loop = tqdm(forward_pass_nums)
    for nforward in loop:
        loop.set_description(f"{nforward=}")
        models = list(
            train_mnist(hidden_features=hidden_features, weight_decay=weight_decay)
            for _ in range(num_models)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        losses = []
        accs = []
        for model in models:
            loss, acc = eval_fn_nfold(model, device, nforward)
            losses.append(loss)
            accs.append(acc)

        loss = sum(losses) / len(losses)
        acc = sum(accs) / len(accs)

        mm = rebasin.MergeMany(
            models,
            MLP(hidden_features=hidden_features).to(device),
            torch.randn(64, 28*28).to(device),
            device=device
        )
        mm.run()

        # Test the merged model
        loss_merged, acc_merged = eval_fn_nfold(mm.merged_model, device, nforward)

        loop.write(f"{loss=}, {loss_merged=}, {acc=}, {acc_merged=}")
        results["nforward"].append(nforward)
        results["loss_avg"].append(loss)
        results["acc_avg"].append(acc)
        results["loss_merged"].append(loss_merged)
        results["acc_merged"].append(acc_merged)

    df = pd.DataFrame(results)
    df.to_csv(
        f"test_merge_many_nfold"
        f"_wd{weight_decay}"
        f"_hf{hidden_features}"
        f"_nm{num_models}.csv",
        index=False
    )


def vec_angle(vec1: torch.nn.Parameter, vec2: torch.nn.Parameter) -> float:
    """Return the angle between two vectors in degrees."""
    return torch.acos(torch.dot(vec1, vec2) / (vec1.norm() * vec2.norm())).item() * 180 / torch.pi


def get_weight_infos(models: list[MLP]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    abs_mean_diffs = []
    eigval_diffs = []
    eigvec_angles = []
    for info in zip(*[model.named_parameters() for model in models]):
        if "weight" not in info[0][0] or info[0][1].ndim != 2:
            continue

        abs_means = torch.tensor([p.abs().mean() for _, p in info])
        abs_mean_diff = (
                (torch.max(abs_means) - torch.min(abs_means))
                / torch.mean(abs_means)  # always positive due to squaring
        )

        abs_mean_diffs.append(abs_mean_diff)

        # Get the eigendecomposition of the weight matrix
        eigvals, eigvecs = [], []
        for _, p in info:
            eigval, eigvec = torch.eig(p, eigenvectors=True)
            eigvals.append(eigval)
            eigvecs.append(eigvec)

        eigval_diff = (
            (torch.max(torch.tensor(eigvals)) - torch.min(torch.tensor(eigvals)))
            / torch.mean(torch.tensor(eigvals))
        )
        eigval_diffs.append(eigval_diff)

        eigvec_angle = torch.mean(
            torch.tensor(
                [vec_angle(v1, v2) for v1, v2 in itertools.combinations(eigvecs, 2)]
            )
        )
        eigvec_angles.append(eigvec_angle)

    return (
        torch.mean(torch.tensor(abs_mean_diffs)),
        torch.mean(torch.tensor(eigval_diffs)),
        torch.mean(torch.tensor(eigvec_angles))
    )


def test_weight_statistics(
        hidden_feature_sizes: Sequence[int],
        weight_decays: Sequence[float],
        model_nums: Sequence[int],
) -> None:
    # Meant to see if filter size corresponds to an implicit regularization
    results = {
        "weight_decay": [],
        "hidden_features": [],
        "num_models": [],
        "abs_mean_diff": [],
        "eigval_diff": [],
        "eigvec_angle": [],
    }

    loop = tqdm(
        itertools.product(weight_decays, hidden_feature_sizes, model_nums),
        total=len(weight_decays) * len(hidden_feature_sizes) * len(model_nums)
    )

    last_settings = {"wd": -1, "hf": -1, "num_models": -1}
    models = []
    for wd, hf, nm in loop:
        loop.set_description(f"{wd=}, {hf=}")
        results["weight_decay"].append(wd)
        results["hidden_features"].append(hf)
        results["num_models"].append(nm)

        # Make it so that I don't have to retrain models if I don't have to
        # (i.e. if the hyperparameters are the same)
        if last_settings["wd"] != wd or last_settings["hf"] != hf:
            last_settings["wd"] = wd
            last_settings["hf"] = hf
            del models  # force the freeing-up of memory
            models = list(
                train_mnist(hidden_features=hf, weight_decay=wd)
                for _ in range(nm)
            )
        elif nm > last_settings["num_models"]:
            models.extend(
                train_mnist(hidden_features=hf, weight_decay=wd)
                for _ in range(nm - last_settings["num_models"])
            )
            last_settings["num_models"] = nm
        elif nm < last_settings["num_models"]:
            models = models[:nm]
            last_settings["num_models"] = nm
        else:
            continue

        abs_mean_diff, eigval_diff, eigvec_angle = get_weight_infos(models)

        results[f"abs_mean_diff"].append(abs_mean_diff)
        results[f"eigval_diff"].append(eigval_diff)
        results[f"eigvec_angle"].append(eigvec_angle)
        loop.write(f"{wd=:.3f}, {hf=:.3f}, {nm=}, {abs_mean_diff=:.3f}, {eigval_diff=:.3f}, {eigvec_angle=:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(
        f"weight_statistics"
        f"_hf{hidden_feature_sizes[0]}-{hidden_feature_sizes[-1]}"
        f"_wd{weight_decays[0]}-{weight_decays[-1]}"
        f"_nm{model_nums[0]}-{model_nums[-1]}.csv",
        index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight_decay', type=float, default=[0.0], nargs='+')
    parser.add_argument('-f', '--hidden_features', type=int, default=[28*28], nargs='+')
    parser.add_argument('-p', '--print_model', action='store_true', default=False)
    parser.add_argument('-s', '--show_permutations', action='store_true', default=False)
    parser.add_argument('-t', '--tune_training_parameters', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-m', '--merge_many', action='store_true', default=False)
    parser.add_argument('-n', '--num_models', type=int, default=[3], nargs='+')
    parser.add_argument('-c', '--count_permutations', action='store_true', default=False)
    parser.add_argument('--full_wd_hf_sweep', action='store_true', default=False)
    parser.add_argument('--compare_output_statistics', action='store_true', default=False)
    parser.add_argument('--forward_pass_nums', type=int, default=None, nargs='+')
    parser.add_argument('--test_weight_statistics', action='store_true', default=False)

    args = parser.parse_args()

    if args.tune_training_parameters:
        tune_training_parameters()
        return

    if args.show_permutations:
        for hf in args.hidden_features:
            show_permutations(hf)
        return

    if args.count_permutations:
        for hf in args.hidden_features:
            count_permutations(hf, args.weight_decay)
        return

    if args.print_model:
        for hf in args.hidden_features:
            print_model(hf)
        return

    if args.full_wd_hf_sweep:
        full_wd_hf_sweep_merge_many()
        return

    if args.compare_output_statistics:
        compare_output_statistics(args.hidden_features, args.weight_decay, args.num_models)
        return

    if args.merge_many:
        test_merge_many(
            args.hidden_features,
            args.weight_decay,
            args.num_models,
            verbose=args.verbose
        )
        return

    if args.forward_pass_nums is not None:
        for hf in args.hidden_features:
            for wd in args.weight_decay:
                for nm in args.num_models:
                    test_merge_many_nfold(hf, wd, nm, args.forward_pass_nums)
        return

    if args.test_weight_statistics:
        test_weight_statistics(
            args.hidden_features,
            args.weight_decay,
            args.num_models,
        )
        return

    for weight_decay in args.weight_decay:
        for hidden_features in args.hidden_features:
            test_permutation_coordinate_descent(
                weight_decay=weight_decay, hidden_features=hidden_features
            )


if __name__ == '__main__':
    main()
