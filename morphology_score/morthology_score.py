#!/usr/bin/env python
"""Apply the fixed RamiGlyph reference morphology-complexity trajectory.

The reference scaler, PCA model, principal curve, and score direction are never
refitted on the input data. The resulting score has the fixed interpretation
``0 = simple branching`` and ``1 = complex branching``.

Expected input layout::

    MyDataset/
    └── val/
        ├── Control/*.swc
        └── Treatment/*.swc

Example::

    python morphology_score_artifacts/morthology_score.py \
        --data-root /path/to/MyDataset \
        --split val \
        --checkpoint morphology_score_artifacts/reference/best_checkpoint.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Batch, Data
from tqdm import tqdm


MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from augmentation.function import (  # noqa: E402
    neighbors_to_adjacency_attr,
    neighbors_to_adjacency_pos,
)
from augmentation.reduce_node import (  # noqa: E402
    compute_branch_id_from_neighbors,
    compute_branch_level_from_neighbors,
    compute_node_distances,
    remap_neighbors,
    subsample_graph,
)
from dataloader.microglia import parse_swc_file  # noqa: E402
from model.dual_branch_model import build_dual_branch_swav_model  # noqa: E402
from model.topo_utils import compute_topological_features_single  # noqa: E402
from score_reporting import write_core_outputs  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "config.json"
DEFAULT_ARTIFACT_DIR = MODULE_ROOT / "reference"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_DIR / "best_checkpoint.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "morphology_score_results"


def set_seed(seed: int) -> None:
    """Seed random number generators used during graph preprocessing."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_config(config: Mapping[str, Any], required: Mapping[str, Any]) -> None:
    """Require all reference-relevant configuration values to match."""
    mismatches = []
    for section, expected_values in required.items():
        observed_values = config.get(section, {})
        for key, expected in expected_values.items():
            observed = observed_values.get(key)
            if observed != expected:
                mismatches.append(
                    f"{section}.{key}: expected={expected!r}, observed={observed!r}"
                )

    if mismatches:
        raise ValueError(
            "Configuration is incompatible with the fixed reference trajectory:\n"
            + "\n".join(mismatches)
        )


def load_reference_artifacts(artifact_dir: Path) -> dict[str, Any]:
    """Load and validate the frozen reference trajectory artifacts."""
    manifest_path = artifact_dir / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Artifact manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    expected_hashes = manifest["artifacts_sha256"]
    artifact_paths = {filename: artifact_dir / filename for filename in expected_hashes}
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reference artifacts:\n" + "\n".join(missing))

    observed_hashes = {
        filename: sha256_file(path) for filename, path in artifact_paths.items()
    }
    mismatches = [
        f"{filename}: expected={expected_hashes[filename]}, "
        f"observed={observed_hashes[filename]}"
        for filename in expected_hashes
        if observed_hashes[filename] != expected_hashes[filename]
    ]
    if mismatches:
        raise RuntimeError(
            "Reference artifact integrity check failed:\n" + "\n".join(mismatches)
        )

    transform_path = artifact_dir / "reference_transform.npz"
    with np.load(transform_path, allow_pickle=False) as transform_file:
        transform = {key: transform_file[key].copy() for key in transform_file.files}

    curve = np.load(
        artifact_dir / "reference_principal_curve.npy", allow_pickle=False
    ).astype(float)
    curve_s = np.load(
        artifact_dir / "reference_curve_arclength.npy", allow_pickle=False
    ).astype(float)
    embedding_dim = int(manifest["embedding_dim"])
    pca_components = int(manifest["pca_components"])
    trajectory_dimensions = int(manifest["trajectory_dimensions"])
    expected_shapes = {
        "scaler_mean": (embedding_dim,),
        "scaler_scale": (embedding_dim,),
        "pca_mean": (embedding_dim,),
        "pca_components": (pca_components, embedding_dim),
    }
    for name, expected_shape in expected_shapes.items():
        if name not in transform or transform[name].shape != expected_shape:
            observed_shape = transform.get(name, np.empty(0)).shape
            raise ValueError(
                f"Invalid {name} shape: expected={expected_shape}, "
                f"observed={observed_shape}"
            )
    if np.any(transform["scaler_scale"] == 0):
        raise ValueError("Reference scaler contains a zero scale value")
    if curve.ndim != 2 or curve.shape[1] != trajectory_dimensions or len(curve) < 2:
        raise ValueError(f"Invalid reference curve shape: {curve.shape}")
    if curve_s.shape != (len(curve),):
        raise ValueError("Reference arclength must contain one value per curve point")
    if not np.all(np.diff(curve_s) >= 0):
        raise ValueError("Reference arclength is not monotonic")
    if not np.isclose(curve_s[0], 0.0) or not np.isclose(curve_s[-1], 1.0):
        raise ValueError("Reference arclength must be normalized from 0 to 1")

    return {
        "manifest": manifest,
        "transform": transform,
        "curve": curve,
        "curve_s": curve_s,
        "hashes": observed_hashes,
    }


def validate_checkpoint(checkpoint_path: Path, manifest: Mapping[str, Any]) -> str:
    """Require the checkpoint used to establish the fixed trajectory."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    observed_hash = sha256_file(checkpoint_path)
    expected_hash = manifest["reference_checkpoint"]["sha256"]
    if observed_hash != expected_hash:
        raise RuntimeError(
            "Checkpoint does not match the fixed reference trajectory.\n"
            f"Expected SHA-256: {expected_hash}\n"
            f"Observed SHA-256: {observed_hash}\n"
            "Use the original reference checkpoint; do not refit or reorient the "
            "trajectory on the evaluation data."
        )
    return observed_hash


def scan_swc_files(data_root: Path, split: str) -> list[dict[str, Any]]:
    """Scan one split and use first-level folder names as group labels."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    category_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not category_dirs:
        raise ValueError(
            f"No group folders found in {split_dir}. Expected '<split>/<group>/*.swc'."
        )

    records = []
    for category_dir in category_dirs:
        for swc_path in sorted(category_dir.rglob("*.swc")):
            records.append(
                {
                    "cell_id": swc_path.stem,
                    "group_label": category_dir.name,
                    "split": split,
                    "source_path": swc_path.relative_to(data_root).as_posix(),
                    "path": swc_path,
                }
            )

    if not records:
        raise ValueError(f"No SWC files found below {split_dir}")
    return records


def swc_to_graph(swc_path: Path, soma_id: int = 0, keep_nodes: int = 1000):
    """Convert one SWC file using the fixed-reference preprocessing policy."""
    features, neighbors = parse_swc_file(swc_path)
    if features is None or neighbors is None or soma_id not in neighbors:
        return None

    neighbors, _ = subsample_graph(
        neighbors=neighbors,
        not_deleted=set(range(len(neighbors))),
        keep_nodes=keep_nodes,
        protected=[soma_id],
        soma_id=soma_id,
    )
    neighbors, old_to_new = remap_neighbors(neighbors, soma_id=soma_id)
    features = features[sorted(old_to_new.keys())]

    node_distances = compute_node_distances(0, neighbors)
    distances = torch.tensor(
        [node_distances[index] for index in range(len(node_distances))],
        dtype=torch.float,
    )
    branch_level = compute_branch_level_from_neighbors(neighbors, soma_id=0)
    branch_id = compute_branch_id_from_neighbors(neighbors, soma_id=0)

    positions = features[:, :3]
    node_types = features[:, 4:]
    centered_features = features.copy()
    centered_features[:, :3] -= centered_features[soma_id, :3].copy()
    x = torch.tensor(centered_features, dtype=torch.float)

    position_min = positions.min()
    position_max = positions.max()
    if position_max == position_min:
        return None
    positions = torch.tensor(
        (positions - position_min) / (position_max - position_min),
        dtype=torch.float,
    )

    adjacency_attr = neighbors_to_adjacency_attr(neighbors, node_types)
    graph = nx.Graph()
    graph.add_nodes_from(neighbors)
    for source, targets in neighbors.items():
        for target in targets:
            if source < target:
                graph.add_edge(source, target)
    if not nx.is_connected(graph):
        return None

    coo_attr = sp.coo_matrix(adjacency_attr)
    edge_index = torch.tensor(np.vstack((coo_attr.row, coo_attr.col)), dtype=torch.long)
    edge_attr = torch.tensor(coo_attr.data, dtype=torch.long)
    coo_position = sp.coo_matrix(neighbors_to_adjacency_pos(neighbors, positions))
    edge_weight = torch.tensor(coo_position.data, dtype=torch.float)
    if edge_index.shape[1] == 0:
        return None

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        pos=positions,
        distance=distances,
        branch_level=branch_level,
        branch_id=branch_id,
        cell_id=swc_path.stem,
    )


def preprocess_record(
    index: int,
    record: Mapping[str, Any],
    topological_config: Mapping[str, Any],
    keep_nodes: int,
):
    """Prepare one graph and persistence image for model inference."""
    try:
        graph = swc_to_graph(
            Path(record["path"]),
            soma_id=int(topological_config.get("soma_node_id", 0)),
            keep_nodes=keep_nodes,
        )
        if graph is None:
            return index, None, None, "invalid or disconnected graph"

        graph = T.AddRandomWalkPE(walk_length=20, attr_name="pe")(graph)
        topology = compute_topological_features_single(
            graph,
            resolution=int(topological_config.get("resolution", 100)),
            soma_node_id=int(topological_config.get("soma_node_id", 0)),
        )
        return index, graph, topology, None
    except Exception as error:
        return index, None, None, f"{type(error).__name__}: {error}"


@torch.inference_mode()
def forward_batch(buffer, model, device):
    """Encode one buffered batch of graphs and topology images."""
    graph_batch = Batch.from_data_list([item[1] for item in buffer]).to(device)
    topology_batch = (
        torch.from_numpy(np.asarray([item[2] for item in buffer]))
        .float()
        .unsqueeze(1)
        .to(device)
    )
    structural_features, topological_features = model.encoder(
        graph_batch, topology_batch
    )
    embeddings = torch.cat([structural_features, topological_features], dim=1).float()
    indices = [item[0] for item in buffer]
    return embeddings.cpu().numpy(), indices


@torch.inference_mode()
def extract_embeddings(
    records,
    model,
    topological_config,
    device,
    batch_size: int,
    keep_nodes: int,
):
    """Stream SWC preprocessing and model inference in bounded batches."""
    embedding_batches = []
    valid_indices = []
    failures = []
    buffer = []

    for index, record in enumerate(tqdm(records, desc="Extracting embeddings")):
        item = preprocess_record(index, record, topological_config, keep_nodes)
        if item[1] is None:
            failures.append(
                {
                    "cell_id": record["cell_id"],
                    "source_path": record["source_path"],
                    "reason": item[3],
                }
            )
            continue

        buffer.append(item)
        if len(buffer) >= batch_size:
            embeddings, indices = forward_batch(buffer, model, device)
            embedding_batches.append(embeddings)
            valid_indices.extend(indices)
            buffer.clear()

    if buffer:
        embeddings, indices = forward_batch(buffer, model, device)
        embedding_batches.append(embeddings)
        valid_indices.extend(indices)

    if not embedding_batches:
        raise RuntimeError("No valid SWC graphs were available for scoring")
    return np.vstack(embedding_batches), valid_indices, failures


def apply_reference_transform(
    embeddings: np.ndarray, transform: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Apply the frozen standardization and PCA transforms."""
    standardized = embeddings.copy()
    standardized -= transform["scaler_mean"]
    standardized /= transform["scaler_scale"]

    pca_coordinates = standardized @ transform["pca_components"].T
    pca_coordinates -= transform["pca_mean"] @ transform["pca_components"].T
    return pca_coordinates


def project_to_curve(
    points: np.ndarray,
    curve: np.ndarray,
    curve_s: np.ndarray,
    chunk_size: int = 512,
):
    """Project points onto all curve segments and return score and residual."""
    segment_start = curve[:-1]
    segment_vector = curve[1:] - curve[:-1]
    denominator = np.maximum(np.sum(segment_vector * segment_vector, axis=1), 1e-15)
    scores = np.empty(len(points), dtype=float)
    residuals = np.empty(len(points), dtype=float)

    for first in range(0, len(points), chunk_size):
        last = min(first + chunk_size, len(points))
        batch = points[first:last]
        relative = batch[:, None, :] - segment_start[None, :, :]
        fraction = (
            np.einsum("bmd,md->bm", relative, segment_vector) / denominator[None, :]
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        projection = (
            segment_start[None, :, :]
            + fraction[:, :, None] * segment_vector[None, :, :]
        )
        distance_squared = np.sum((batch[:, None, :] - projection) ** 2, axis=2)
        best_segment = np.argmin(distance_squared, axis=1)
        row_indices = np.arange(len(batch))
        best_fraction = fraction[row_indices, best_segment]
        scores[first:last] = curve_s[best_segment] + best_fraction * (
            curve_s[best_segment + 1] - curve_s[best_segment]
        )
        residuals[first:last] = np.sqrt(distance_squared[row_indices, best_segment])

    return np.clip(scores, 0.0, 1.0), residuals


def reference_residual_threshold(manifest: Mapping[str, Any]) -> float:
    """Read the fixed training-set P99 trajectory residual threshold."""
    value = manifest.get("reference_residual_p99")
    if value is None or not np.isfinite(float(value)):
        raise KeyError("No valid reference_residual_p99 is stored in the manifest")
    return float(value)


def build_score_table(
    records,
    valid_indices,
    embeddings,
    reference,
    endpoint_margin: float,
    projection_chunk_size: int,
):
    """Build cell-level fixed-reference morphology scores."""
    expected_dim = int(reference["manifest"]["embedding_dim"])
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Expected embeddings with shape (n, {expected_dim}), got {embeddings.shape}"
        )

    pca_coordinates = apply_reference_transform(embeddings, reference["transform"])
    trajectory_dimensions = int(reference["manifest"]["trajectory_dimensions"])
    scores, residuals = project_to_curve(
        pca_coordinates[:, :trajectory_dimensions],
        reference["curve"],
        reference["curve_s"],
        chunk_size=projection_chunk_size,
    )
    threshold = reference_residual_threshold(reference["manifest"])

    rows = []
    for output_index, record_index in enumerate(valid_indices):
        record = records[record_index]
        score = float(scores[output_index])
        residual = float(residuals[output_index])
        rows.append(
            {
                "cell_id": record["cell_id"],
                "group_label": record["group_label"],
                "split": record["split"],
                "source_path": record["source_path"],
                "morphology_complexity_score": score,
                "trajectory_residual": residual,
                "reference_high_residual_threshold": threshold,
                "high_trajectory_residual": residual > threshold,
                "lower_endpoint_saturation": score <= endpoint_margin,
                "upper_endpoint_saturation": score >= 1.0 - endpoint_margin,
                "trajectory_source": "fixed_reference_principal_curve",
            }
        )
    return pd.DataFrame(rows)


def load_model(config, checkpoint_path: Path, device):
    """Load the frozen encoder required by the reference trajectory."""
    model = build_dual_branch_swav_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keep-nodes", type=int, default=1000)
    parser.add_argument("--projection-chunk-size", type=int, default=512)
    parser.add_argument("--endpoint-margin", type=float, default=0.01)
    parser.add_argument("--density-bandwidth", type=float, default=0.035)
    parser.add_argument("--density-grid-points", type=int, default=501)
    parser.add_argument(
        "--group-order",
        nargs="+",
        default=None,
        help="Optional display order containing every observed group label",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Extract embeddings and apply the frozen morphology trajectory."""
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.keep_nodes < 1 or args.projection_chunk_size < 1:
        raise ValueError(
            "Batch size, keep nodes, and projection chunk size must be positive"
        )
    if not 0.0 <= args.endpoint_margin < 0.5:
        raise ValueError("Endpoint margin must be in [0, 0.5)")
    if args.density_bandwidth <= 0:
        raise ValueError("Density bandwidth must be positive")
    if args.density_grid_points < 2:
        raise ValueError("Density grid points must be at least 2")

    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    reference = load_reference_artifacts(artifact_dir)
    config = load_json(config_path)
    validate_config(config, reference["manifest"]["required_config"])
    validate_checkpoint(checkpoint_path, reference["manifest"])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    records = scan_swc_files(data_root, args.split)
    print(f"Found {len(records)} SWC files in {data_root / args.split}")
    print(f"Using device: {device}")

    model = load_model(config, checkpoint_path, device)
    embeddings, valid_indices, failures = extract_embeddings(
        records,
        model,
        config.get("topological", {}),
        device,
        batch_size=args.batch_size,
        keep_nodes=args.keep_nodes,
    )
    score_table = build_score_table(
        records,
        valid_indices,
        embeddings,
        reference,
        endpoint_margin=args.endpoint_margin,
        projection_chunk_size=args.projection_chunk_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "morphology_scores.csv"
    failure_path = output_dir / "failed_samples.csv"

    score_table.to_csv(score_path, index=False)
    write_core_outputs(
        score_table,
        output_dir,
        requested_group_order=args.group_order,
        density_bandwidth=args.density_bandwidth,
        density_grid_points=args.density_grid_points,
    )
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)

    print(f"Scored cells: {len(score_table)} / {len(records)}")
    print(f"Morphology scores saved to: {score_path}")
    print(f"Group summary saved to: {output_dir / 'group_summary.csv'}")
    print(f"Score density saved to: {output_dir / 'score_density_all_groups.svg'}")
    print(
        "Trajectory residuals saved to: "
        f"{output_dir / 'trajectory_residual_by_group.svg'}"
    )


if __name__ == "__main__":
    main()
