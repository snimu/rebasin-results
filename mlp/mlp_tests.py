"""Train an MLP on MNIST."""
from __future__ import annotations

import argparse
import copy
import os
import itertools
import shutil
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
            in_features: int,
            out_features: int,
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
            loop.write("Create models")
        models = []
        for i in range(model_num):
            models.append(
                train_mnist(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    hidden_features=feature_num,
                    verbose=verbose
                ).to(device)
            )

        x = torch.randn(64, 28*28).to(device)

        if verbose:
            loop.write("Evaluate models")

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
            loop.write("Merge models")
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
            loop.write(f"Merged model accuracy: {acc_merged}")

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

    args = parser.parse_args()

    if args.tune_training_parameters:
        tune_training_parameters()
        return

    if args.show_permutations:
        for hf in args.hidden_features:
            show_permutations(hf)
        return

    if args.print_model:
        for hf in args.hidden_features:
            print_model(hf)
        return

    if args.merge_many:
        test_merge_many(
            args.hidden_features,
            args.weight_decay,
            args.num_models,
            verbose=args.verbose
        )
        return

    for weight_decay in args.weight_decay:
        for hidden_features in args.hidden_features:
            test_permutation_coordinate_descent(
                weight_decay=weight_decay, hidden_features=hidden_features
            )


if __name__ == '__main__':
    main()
