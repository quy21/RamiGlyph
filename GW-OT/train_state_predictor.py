#!/usr/bin/env python3
"""Train the morphology-to-functional-state probability model.

The input CSV must contain morphology embeddings (e000, e001, ...), condition
and region annotations, eight GW-OT state targets named ``target_state_*``,
159 pathway targets named ``target_pathway_z_*``, and ``dominant_state``.

Usage:
    python GW-OT/train_state_predictor.py \
        --input-csv path/to/dsbm_pathway_gw_targets.csv \
        --output-dir outputs/state_prediction
"""

import argparse
import copy
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


STATE_NAMES = [
    "Antigen-presenting_MG",
    "IFN-responsive_MG",
    "Inflammatory_MG",
    "Migratory-chemotactic_MG",
    "Neuroprotective_MG",
    "Phagolysosomal_MG",
    "Proliferative_MG",
    "Surveillance_MG",
]


class SwiGLUBlock(nn.Module):
    """Pre-normalized gated feed-forward residual block."""

    def __init__(self, hidden_dim, dropout, expansion, residual_scale):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.input = nn.Linear(hidden_dim, hidden_dim * expansion * 2)
        self.output = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, values):
        value, gate = self.input(self.norm(values)).chunk(2, dim=-1)
        update = self.output(self.dropout(F.silu(gate) * value))
        return values + self.scale * self.dropout(update)


class MorphologyStateNet(nn.Module):
    """Predict functional-state probabilities and pathway activities."""

    def __init__(
        self,
        input_dim,
        pathway_dim,
        hidden_dim=512,
        blocks=3,
        dropout=0.1,
        input_dropout=0.05,
        expansion=2,
        residual_scale=0.1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_gate = nn.Parameter(torch.zeros(input_dim))
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU()
        )
        self.blocks = nn.Sequential(
            *[
                SwiGLUBlock(
                    hidden_dim, dropout, expansion, residual_scale
                )
                for _ in range(blocks)
            ]
        )
        self.state_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(STATE_NAMES))
        )
        self.pathway_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, pathway_dim)
        )

    def forward(self, morphology):
        gate = 2.0 * torch.sigmoid(self.feature_gate)
        hidden = self.input_dropout(self.input_norm(morphology) * gate)
        hidden = self.blocks(self.input_projection(hidden))
        return self.state_head(hidden), self.pathway_head(hidden)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the morphology-to-state probability model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input-dropout", type=float, default=0.05)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--pathway-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def load_data(path):
    frame = pd.read_csv(path)
    morphology_columns = sorted(
        [column for column in frame if re.fullmatch(r"e\d{3}", column)]
    )
    pathway_columns = sorted(
        [column for column in frame if re.fullmatch(r"target_pathway_z_\d+", column)]
    )
    state_columns = [f"target_state_{state}" for state in STATE_NAMES]
    required = ["condition", "region", "dominant_state", *state_columns]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"Input CSV is missing columns: {missing}")
    if not morphology_columns or not pathway_columns:
        raise ValueError("Morphology or pathway target columns were not found.")

    state_index = {state: index for index, state in enumerate(STATE_NAMES)}
    unknown = sorted(set(frame["dominant_state"].astype(str)) - set(state_index))
    if unknown:
        raise ValueError(f"Unknown dominant states: {unknown}")
    return {
        "morphology": frame[morphology_columns].to_numpy(np.float32),
        "pathway": frame[pathway_columns].to_numpy(np.float32),
        "state_score": frame[state_columns].to_numpy(np.float32),
        "hard_state": frame["dominant_state"].map(state_index).to_numpy(np.int64),
        "condition": frame["condition"].astype(str).to_numpy(),
        "region": frame["region"].astype(str).to_numpy(),
    }


def make_folds(labels, folds, seed):
    counts = pd.Series(labels).value_counts()
    indices = np.arange(len(labels))
    if len(counts) and counts.min() >= folds:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        return splitter.split(indices, labels)
    return KFold(folds, shuffle=True, random_state=seed).split(indices)


def soft_targets(scores, mean, standard_deviation, temperature):
    standardized = (scores - mean) / standard_deviation
    return F.softmax(
        torch.tensor(standardized / temperature, dtype=torch.float32), dim=1
    ).numpy()


def make_loader(x, soft, hard, pathway, batch_size, shuffle):
    dataset = TensorDataset(
        torch.tensor(x),
        torch.tensor(soft),
        torch.tensor(hard),
        torch.tensor(pathway),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def combined_loss(outputs, targets, pathway_weight):
    logits, pathway_prediction = outputs
    soft, hard, pathway = targets
    return (
        F.cross_entropy(logits, hard)
        + F.kl_div(F.log_softmax(logits, dim=1), soft, reduction="batchmean")
        + pathway_weight * F.mse_loss(pathway_prediction, pathway)
    )


def validation_loss(model, loader, device, pathway_weight):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, soft, hard, pathway in loader:
            x, soft = x.to(device), soft.to(device)
            hard, pathway = hard.to(device), pathway.to(device)
            loss = combined_loss(
                model(x), (soft, hard, pathway), pathway_weight
            )
            total += float(loss) * len(x)
    return total / len(loader.dataset)


def predict(model, values, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.tensor(
                values[start : start + batch_size], device=device
            )
            logits, _ = model(batch)
            predictions.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(predictions)


def train_fold(data, train_index, test_index, fold, args, device):
    set_seed(args.seed)
    scaler = StandardScaler().fit(data["morphology"][train_index])
    x = scaler.transform(data["morphology"]).astype(np.float32)
    state_mean = data["state_score"][train_index].mean(0)
    state_std = data["state_score"][train_index].std(0) + 1e-8
    soft = soft_targets(
        data["state_score"], state_mean, state_std, args.temperature
    )

    fit_index, validation_index = train_test_split(
        train_index,
        test_size=args.validation_fraction,
        random_state=args.seed,
        stratify=data["hard_state"][train_index],
    )
    train_loader = make_loader(
        x[fit_index],
        soft[fit_index],
        data["hard_state"][fit_index],
        data["pathway"][fit_index],
        args.batch_size,
        True,
    )
    validation_loader = make_loader(
        x[validation_index],
        soft[validation_index],
        data["hard_state"][validation_index],
        data["pathway"][validation_index],
        args.batch_size,
        False,
    )
    model = MorphologyStateNet(
        x.shape[1],
        data["pathway"].shape[1],
        args.hidden_dim,
        args.blocks,
        args.dropout,
        args.input_dropout,
        args.expansion,
        args.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.05
    )

    best_state, best_loss, stale_epochs = None, np.inf, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_x, batch_soft, batch_hard, batch_pathway in train_loader:
            batch_x, batch_soft = batch_x.to(device), batch_soft.to(device)
            batch_hard = batch_hard.to(device)
            batch_pathway = batch_pathway.to(device)
            loss = combined_loss(
                model(batch_x),
                (batch_soft, batch_hard, batch_pathway),
                args.pathway_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        current_loss = validation_loss(
            model, validation_loader, device, args.pathway_weight
        )
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"fold={fold} epoch={epoch} validation_loss={current_loss:.5f}")
        if stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    probability = predict(model, x[test_index], args.batch_size, device)
    prediction = probability.argmax(1)
    target = data["hard_state"][test_index]
    top_two = np.argsort(probability, axis=1)[:, -2:]
    metrics = {
        "fold": fold,
        "cell_accuracy": accuracy_score(target, prediction),
        "cell_balanced_accuracy": balanced_accuracy_score(target, prediction),
        "cell_macro_f1": f1_score(target, prediction, average="macro"),
        "cell_top2_accuracy": np.mean(
            [truth in choices for truth, choices in zip(target, top_two)]
        ),
    }
    checkpoint = {
        "model": model.state_dict(),
        "morphology_mean": scaler.mean_,
        "morphology_scale": scaler.scale_,
        "state_names": STATE_NAMES,
        "state_mean": state_mean,
        "state_std": state_std,
        "config": vars(args),
    }
    torch.save(checkpoint, args.output_dir / f"fold_{fold}.pt")
    return probability, soft[test_index], metrics


def main():
    args = parse_args()
    if args.folds < 2 or args.temperature <= 0:
        raise ValueError("folds must be at least two and temperature must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = choose_device(args.device)
    data = load_data(args.input_csv)
    stratification = np.char.add(
        np.char.add(data["condition"].astype(str), "__"),
        data["hard_state"].astype(str),
    )

    n_cells = len(data["hard_state"])
    state_probability = np.zeros((n_cells, len(STATE_NAMES)), np.float32)
    target_soft_state = np.zeros_like(state_probability)
    fold_assignment = np.full(n_cells, -1, np.int64)
    fold_results = []
    folds = make_folds(stratification, args.folds, args.seed)
    for fold, (train_index, test_index) in enumerate(folds):
        probability, soft_target, metrics = train_fold(
            data, train_index, test_index, fold, args, device
        )
        state_probability[test_index] = probability
        target_soft_state[test_index] = soft_target
        fold_assignment[test_index] = fold
        fold_results.append(metrics)

    np.savez_compressed(
        args.output_dir / "oof_predictions.npz",
        state_prob=state_probability,
        target_soft_state=target_soft_state,
        target_hard_state=data["hard_state"],
        condition=data["condition"],
        region=data["region"],
        fold=fold_assignment,
        state_names=np.asarray(STATE_NAMES),
    )
    pd.DataFrame(fold_results).to_csv(
        args.output_dir / "fold_results.csv", index=False
    )
    with (args.output_dir / "run_parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(vars(args), handle, indent=2, default=str)
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
