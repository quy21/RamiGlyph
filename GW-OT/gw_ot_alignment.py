#!/usr/bin/env python3
"""Condition-wise GW-OT alignment of morphology and scRNA-seq activity.

This public script follows the analysis used in RamiGlyph: cosine geometry for
morphology embeddings, Euclidean geometry for standardized AUCell pathway
profiles, uniform marginals, and barycentric transfer of eight state scores.

Usage:
    python GW-OT/gw_ot_alignment.py \
        --morphology-csv path/to/embeddings.csv \
        --pathway-csv path/to/auc_cell_x_pathway.csv \
        --state-csv path/to/auc_cell_x_category.csv \
        --output-dir outputs/gw_ot
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

try:
    import ot
except ImportError:
    ot = None


STATE_COLUMNS = [
    "Antigen-presenting_MG",
    "IFN-responsive_MG",
    "Inflammatory_MG",
    "Migratory-chemotactic_MG",
    "Neuroprotective_MG",
    "Phagolysosomal_MG",
    "Proliferative_MG",
    "Surveillance_MG",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align morphology embeddings and scRNA-seq AUCell profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--morphology-csv", type=Path, required=True)
    parser.add_argument("--pathway-csv", type=Path, required=True)
    parser.add_argument("--state-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["Control", "Rep-1d", "Rep-3d", "Rep-5d", "Rep-7d"],
    )
    parser.add_argument(
        "--transcript-condition-regex",
        default=r"-(Control|Rep-\d+d)$",
        help="Regex whose first capture group is the condition.",
    )
    parser.add_argument("--epsilon", type=float, default=5e-3)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--initializations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def column_zscore(values, epsilon=1e-9):
    """Standardize each feature across cells."""
    return (values - values.mean(0, keepdims=True)) / (
        values.std(0, keepdims=True) + epsilon
    )


def load_morphology(path):
    """Load metadata and e000/e001/... morphology embedding columns."""
    frame = pd.read_csv(path)
    embedding_columns = [column for column in frame if re.fullmatch(r"e\d{3}", column)]
    if not embedding_columns:
        raise ValueError("No morphology columns matching e000, e001, ... were found.")
    if "condition" not in frame:
        raise KeyError("Morphology CSV must contain a 'condition' column.")
    embeddings = frame[embedding_columns].to_numpy(dtype=np.float32)
    metadata = frame.drop(columns=embedding_columns).reset_index(drop=True)
    return metadata, embeddings


def load_transcriptomics(state_path, pathway_path, condition_regex):
    """Load matched state and pathway AUCell matrices."""
    states = pd.read_csv(state_path, index_col=0)
    pathways = pd.read_csv(pathway_path, index_col=0)
    cells = states.index.intersection(pathways.index, sort=False)
    if cells.empty:
        raise ValueError("State and pathway tables have no shared cell IDs.")

    states, pathways = states.loc[cells], pathways.loc[cells]
    missing = [column for column in STATE_COLUMNS if column not in states]
    if missing:
        raise KeyError(f"State table is missing columns: {missing}")
    state_values = states[STATE_COLUMNS].to_numpy(dtype=np.float32)
    pathway_values = column_zscore(pathways.to_numpy(dtype=np.float32))
    if not np.isfinite(state_values).all() or not np.isfinite(pathway_values).all():
        raise ValueError("Input activity tables contain non-finite values.")

    conditions = cells.astype(str).str.extract(condition_regex, expand=False)
    if isinstance(conditions, pd.DataFrame):
        conditions = conditions.iloc[:, 0]
    if pd.isna(conditions).any():
        raise ValueError(
            "Some transcriptomic cell IDs do not match the condition regex."
        )
    metadata = pd.DataFrame(
        {"cell_id": cells.astype(str), "condition": np.asarray(conditions)}
    )
    return metadata, state_values, pathway_values


def normalized_distances(values, metric):
    """Return a symmetric within-domain distance matrix scaled to [0, 1]."""
    if metric == "cosine":
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        distances = (1.0 - values @ values.T).clamp(min=0.0)
    else:
        distances = torch.cdist(values, values, p=2)
    distances = distances / distances.max().clamp(min=1e-12)
    return 0.5 * (distances + distances.T)


def condition_seed(base_seed, condition):
    """Create a stable condition-specific seed."""
    digest = hashlib.blake2b(condition.encode("utf-8"), digest_size=4).digest()
    return base_seed + int.from_bytes(digest, "little") % 100_000


def random_coupling(p, q, seed, iterations=30):
    """Generate an approximately feasible random initialization."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    coupling = torch.rand((len(p), len(q)), generator=generator, dtype=p.dtype).to(
        p.device
    )
    for _ in range(iterations):
        coupling *= p[:, None] / coupling.sum(1, keepdim=True).clamp(min=1e-30)
        coupling *= q[None, :] / coupling.sum(0, keepdim=True).clamp(min=1e-30)
    return coupling


def solve_gw(morphology_cost, transcriptomic_cost, p, q, args, seed):
    """Run entropic GW from several starts and retain the lowest-loss solution."""
    if ot is None:
        raise ImportError("Install Python Optimal Transport with `pip install POT`.")
    best_coupling, best_loss, losses = None, np.inf, []
    for initialization in range(args.initializations):
        initial = None
        if initialization:
            initial = random_coupling(p, q, seed + 1000 * initialization)
        coupling, log = ot.gromov.entropic_gromov_wasserstein(
            morphology_cost,
            transcriptomic_cost,
            p,
            q,
            loss_fun="square_loss",
            epsilon=args.epsilon,
            G0=initial,
            solver="PPA",
            max_iter=args.max_iterations,
            tol=args.tolerance,
            log=True,
            verbose=False,
        )
        loss = float(log["gw_dist"])
        losses.append(loss)
        print(f"    start {initialization + 1}/{args.initializations}: loss={loss:.8f}")
        if loss < best_loss:
            best_coupling, best_loss = coupling.detach().clone(), loss
    return best_coupling, best_loss, losses


def barycentric_projection(coupling, activities):
    """Transfer transcriptomic activities through the row-normalized coupling."""
    return (coupling @ activities) / coupling.sum(1, keepdim=True).clamp(min=1e-30)


def coupling_diagnostics(coupling, best_loss, losses):
    """Calculate marginal errors and normalized coupling entropy."""
    n_morphology, n_transcriptomic = coupling.shape
    row_target = np.full(n_morphology, 1.0 / n_morphology)
    column_target = np.full(n_transcriptomic, 1.0 / n_transcriptomic)
    probability = coupling / max(coupling.sum(), 1e-30)
    entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum()
    return {
        "n_morphology": n_morphology,
        "n_transcriptomic": n_transcriptomic,
        "best_gw_loss": best_loss,
        "initialization_losses": losses,
        "maximum_row_marginal_error": float(
            np.abs(coupling.sum(1) - row_target).max()
        ),
        "maximum_column_marginal_error": float(
            np.abs(coupling.sum(0) - column_target).max()
        ),
        "normalized_entropy": float(entropy / np.log(coupling.size)),
    }


def state_proportions(frame):
    """Calculate dominant-state proportions per condition."""
    return (
        frame.groupby("condition")["dominant_state"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=STATE_COLUMNS, fill_value=0)
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    print(f"Using {device}")

    morph_meta, morph_embeddings = load_morphology(args.morphology_csv)
    rna_meta, state_activities, pathway_activities = load_transcriptomics(
        args.state_csv, args.pathway_csv, args.transcript_condition_regex
    )

    projected_tables, diagnostics = [], {}
    for condition in args.conditions:
        morph_idx = np.flatnonzero(morph_meta["condition"].astype(str) == condition)
        rna_idx = np.flatnonzero(rna_meta["condition"].astype(str) == condition)
        if len(morph_idx) == 0 or len(rna_idx) == 0:
            raise ValueError(f"Condition {condition!r} is missing from one modality.")
        print(f"{condition}: morphology={len(morph_idx)}, scRNA-seq={len(rna_idx)}")

        morphology = torch.tensor(
            morph_embeddings[morph_idx], dtype=torch.float64, device=device
        )
        pathways = torch.tensor(
            pathway_activities[rna_idx], dtype=torch.float64, device=device
        )
        states = torch.tensor(
            state_activities[rna_idx], dtype=torch.float64, device=device
        )
        morphology_cost = normalized_distances(morphology, "cosine")
        transcriptomic_cost = normalized_distances(pathways, "euclidean")
        p = torch.full(
            (len(morphology),),
            1.0 / len(morphology),
            dtype=torch.float64,
            device=device,
        )
        q = torch.full(
            (len(pathways),), 1.0 / len(pathways), dtype=torch.float64, device=device
        )
        coupling, best_loss, losses = solve_gw(
            morphology_cost,
            transcriptomic_cost,
            p,
            q,
            args,
            condition_seed(args.seed, condition),
        )

        projected_states = barycentric_projection(coupling, states).cpu().numpy()
        coupling_array = coupling.cpu().numpy().astype(np.float32)
        np.savez_compressed(
            args.output_dir / f"coupling_{condition}.npz",
            T=coupling_array,
            morph_cell_idx=morph_idx,
            scrna_cell_id=rna_meta.iloc[rna_idx]["cell_id"].to_numpy(),
        )
        diagnostics[condition] = coupling_diagnostics(
            coupling_array, best_loss, losses
        )

        projected = morph_meta.iloc[morph_idx].copy()
        projected["_orig_idx"] = morph_idx
        for index, column in enumerate(STATE_COLUMNS):
            projected[column] = projected_states[:, index]
        projected_tables.append(projected)

        del morphology, pathways, states, morphology_cost, transcriptomic_cost, coupling
        if device.type == "cuda":
            torch.cuda.empty_cache()

    morphology_states = pd.concat(projected_tables, ignore_index=True)
    morphology_z = column_zscore(morphology_states[STATE_COLUMNS].to_numpy())
    for index, column in enumerate(STATE_COLUMNS):
        morphology_states[f"{column}_z"] = morphology_z[:, index]
    morphology_states["dominant_state"] = np.asarray(STATE_COLUMNS)[
        morphology_z.argmax(1)
    ]
    morphology_states.to_csv(
        args.output_dir / "morph_with_states.csv", index=False
    )

    transcript_states = rna_meta.copy()
    transcript_z = column_zscore(state_activities)
    transcript_states["dominant_state"] = np.asarray(STATE_COLUMNS)[
        transcript_z.argmax(1)
    ]
    morphology_proportions = state_proportions(morphology_states)
    transcript_proportions = state_proportions(transcript_states)
    morphology_proportions.to_csv(
        args.output_dir / "morph_proportion_per_condition.csv"
    )
    transcript_proportions.to_csv(
        args.output_dir / "scrna_proportion_per_condition.csv"
    )

    validation = {}
    for condition in args.conditions:
        correlation, p_value = pearsonr(
            morphology_proportions.loc[condition],
            transcript_proportions.loc[condition],
        )
        validation[f"per_cond__{condition}__vs__{condition}"] = {
            "r": float(correlation),
            "p": float(p_value),
        }

    control = "Control"
    if control in morphology_proportions.index:
        for condition in args.conditions:
            if condition == control:
                continue
            morph_shift = (
                morphology_proportions.loc[condition]
                - morphology_proportions.loc[control]
            )
            transcript_shift = (
                transcript_proportions.loc[condition]
                - transcript_proportions.loc[control]
            )
            correlation, p_value = pearsonr(morph_shift, transcript_shift)
            validation[f"shift__{condition}__minus__{control}"] = {
                "r": float(correlation),
                "p": float(p_value),
            }

    for filename, content in [
        ("sanity_metrics.json", diagnostics),
        ("validation_metrics.json", validation),
        ("run_parameters.json", vars(args)),
    ]:
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, default=str)
    print(f"Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
