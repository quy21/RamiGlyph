#!/usr/bin/env python
"""Run bootstrap analysis on RamiGlyph representations.

Organize SWC files by class label within the split to analyze::

    MyDataset/
    └── val/
        ├── Control/*.swc
        └── Treatment/*.swc

The first-level folder below the selected split is used as the group label.
Bootstrap calculations use only ``--split``. When processed files do not yet
exist, the current dataset preprocessor still expects both ``train`` and ``val``
directories to be present. Example::

    python evaluation/bootstrap.py \
        --data_root /path/to/MyDataset \
        --split val \
        --checkpoint checkpoints/best_checkpoint.pt
"""

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch_geometric.data import Batch
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloader.microglia import MicrogliaDataset
from evaluation.data_utils import load_labeled_graphs, print_label_distribution
from model.dual_branch_model import build_dual_branch_swav_model
from model.topo_utils import compute_topological_features_single


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.json")
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_checkpoint.pt")

REFERENCE_PALETTE_20 = [
    "#4E79A7",
    "#A0CBE8",
    "#F28E2B",
    "#FFBE7D",
    "#59A14F",
    "#8CD17D",
    "#B6992D",
    "#F1CE63",
    "#E15759",
    "#FF9D9A",
    "#76B7B2",
    "#9D7660",
    "#B07AA1",
    "#D4A6C8",
    "#86BCB6",
    "#D7B5A6",
    "#6B6B6B",
    "#BAB0AC",
    "#D37295",
    "#FABFD2",
]
REFERENCE_PRIMARY_COLORS = REFERENCE_PALETTE_20[::2]


bootstrap_methods = {
    "mean": lambda arr: np.mean(arr),
    "median": lambda arr: np.median(arr),
    "max": lambda arr: np.amax(arr),
    "min": lambda arr: np.amin(arr),
    "mean_axis": lambda arr, ax: np.mean(arr, axis=ax),
    "median_axis": lambda arr, ax: np.median(arr, axis=ax),
}


def _bootstrap_feature(
    bootstrap_bag, feature_type, bootstrap_method=bootstrap_methods["mean_axis"]
):
    if feature_type == "bars":
        bootstrapped_feature = bootstrap_bag
    elif feature_type == "array":
        bootstrapped_feature = bootstrap_method(bootstrap_bag, 0)
    else:
        bootstrapped_feature = bootstrap_method(bootstrap_bag)

    return bootstrapped_feature


def get_bootstrap_frame(
    features_dict,
    N_bags=50,
    n_samples=15,
    replacement=True,
    ratio=None,
    rand_seed=None,
    feature_type="array",
):
    np.random.seed(rand_seed)

    bootstrap_frame_list = []

    for condition, features in features_dict.items():
        print(f"Performing bootstrapping for {condition}...")

        pop_length = len(features)
        print(f"  There are {pop_length} morphologies to bootstrap...")

        if ratio is not None and ratio > 0:
            _n_samples = int(ratio * pop_length)
            if _n_samples == 0:
                _n_samples = 1
        else:
            _n_samples = n_samples

        if not replacement and _n_samples > pop_length:
            _n_samples = pop_length

        sampled_idxs_list = [
            np.random.choice(pop_length, size=_n_samples, replace=replacement)
            for _ in range(N_bags)
        ]

        bootstraped_bag_list = []
        for sampled_idxs in sampled_idxs_list:
            bootstrap_bag = features[sampled_idxs]
            bootstraped_bag = _bootstrap_feature(bootstrap_bag, feature_type)
            bootstraped_bag_list.append(bootstraped_bag)

        condition_bootstrap_frame = pd.DataFrame(
            {
                "condition": [condition] * N_bags,
                "bootstrap_indices": sampled_idxs_list,
                "features": bootstraped_bag_list,
            }
        )

        bootstrap_frame_list.append(condition_bootstrap_frame)

    bootstrap_frame = pd.concat(bootstrap_frame_list, axis=0, ignore_index=True)

    return bootstrap_frame


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def load_model(config, checkpoint_path, device):
    model = build_dual_branch_swav_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"  - Step: {checkpoint.get('step', 'N/A')}")
    print(f"  - Min Loss: {checkpoint.get('min_loss', 'N/A'):.6f}")

    return model


def build_neighbors_from_edge_index(edge_index, num_nodes):
    neighbors = {i: set() for i in range(num_nodes)}
    edge_index_np = edge_index.numpy() if torch.is_tensor(edge_index) else edge_index
    for i in range(edge_index_np.shape[1]):
        src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
        neighbors[src].add(dst)
        neighbors[dst].add(src)
    return neighbors


def compute_branch_level_for_eval(data):
    if hasattr(data, "branch_level") and data.branch_level is not None:
        return data.branch_level

    num_nodes = data.x.size(0)
    neighbors = build_neighbors_from_edge_index(data.edge_index, num_nodes)

    soma_id = 0
    levels = {soma_id: 0}
    visited = {soma_id}
    queue = [(soma_id, -1, 0)]

    while queue:
        node, parent, current_level = queue.pop(0)
        children = [n for n in neighbors[node] if n not in visited]

        if len(children) > 1:
            next_level = current_level + 1
        else:
            next_level = current_level

        for child in children:
            levels[child] = next_level
            visited.add(child)
            queue.append((child, node, next_level))

    level_array = np.zeros(num_nodes)
    for node_id, level in levels.items():
        level_array[node_id] = level

    return torch.tensor(level_array).long()


def compute_branch_id_for_eval(data):
    if hasattr(data, "branch_id") and data.branch_id is not None:
        return data.branch_id

    num_nodes = data.x.size(0)
    neighbors = build_neighbors_from_edge_index(data.edge_index, num_nodes)

    soma_id = 0
    branch_ids = {soma_id: 0}
    visited = {soma_id}

    soma_children = [n for n in neighbors[soma_id]]

    for branch_idx, start_node in enumerate(soma_children, start=1):
        queue = [start_node]
        visited.add(start_node)
        branch_ids[start_node] = branch_idx

        while queue:
            node = queue.pop(0)
            children = [n for n in neighbors[node] if n not in visited]

            for child in children:
                branch_ids[child] = branch_idx
                visited.add(child)
                queue.append(child)

    branch_id_array = np.zeros(num_nodes)
    for node_id, bid in branch_ids.items():
        branch_id_array[node_id] = bid

    return torch.tensor(branch_id_array).long()


@torch.no_grad()
def extract_features(model, data_list, topo_config, device, batch_size=32):
    model.eval()
    pe_transform = T.AddRandomWalkPE(walk_length=20, attr_name="pe")
    all_features = []
    valid_indices = []

    for i in tqdm(range(0, len(data_list), batch_size), desc="Extracting features"):
        batch_data = data_list[i : i + batch_size]
        processed_data = []
        topo_images = []
        batch_valid_indices = []

        for j, data in enumerate(batch_data):
            data_idx = i + j
            num_nodes = data.x.size(0)
            edge_index = data.edge_index

            if edge_index.numel() > 0:
                max_idx = edge_index.max().item()
                if max_idx >= num_nodes:
                    continue

            try:
                data = pe_transform(data)
            except Exception as e:
                print(f"Warning: PE transform failed, skipping: {e}")
                continue

            if not hasattr(data, "branch_level") or data.branch_level is None:
                data.branch_level = compute_branch_level_for_eval(data)
            if not hasattr(data, "branch_id") or data.branch_id is None:
                data.branch_id = compute_branch_id_for_eval(data)

            processed_data.append(data)
            batch_valid_indices.append(data_idx)

            resolution = topo_config.get("resolution", 50)
            soma_node_id = topo_config.get("soma_node_id", 0)
            topo_img = compute_topological_features_single(
                data, resolution, soma_node_id
            )
            topo_images.append(topo_img)

        if len(processed_data) == 0:
            continue

        graph_batch = Batch.from_data_list(processed_data).to(device)
        topo_batch = torch.FloatTensor(np.array(topo_images)).unsqueeze(1).to(device)

        h_struct, h_topo = model.encoder(graph_batch, topo_batch)
        features = torch.cat([h_struct, h_topo], dim=1)

        all_features.append(features.cpu().numpy())
        valid_indices.extend(batch_valid_indices)

    return np.vstack(all_features), valid_indices


def apply_plot_style():
    """Apply the reference Bootstrap figure style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure_svg(fig, path):
    """Save one SVG with editable text."""
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_colors(labels):
    """Assign the reference palette while supporting arbitrary labels."""
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= len(REFERENCE_PRIMARY_COLORS):
        colors = REFERENCE_PRIMARY_COLORS
    elif len(unique_labels) <= len(REFERENCE_PALETTE_20):
        colors = REFERENCE_PALETTE_20
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    return unique_labels, {
        label: colors[index] for index, label in enumerate(unique_labels)
    }


def draw_embedding(coordinates, labels, dataset_name, split, method_name, save_path):
    """Draw a t-SNE or UMAP embedding using the reference figure layout."""
    unique_labels, color_map = label_colors(labels)
    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="white")
    ax.set_facecolor("white")

    for label in unique_labels:
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            c=[color_map[label]],
            label=str(label).replace("_", " "),
            alpha=0.35,
            s=20,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.set_xlabel(f"{method_name} 1", fontsize=11)
    ax.set_ylabel(f"{method_name} 2", fontsize=11)
    ax.set_title(
        f"{method_name} Bootstrap Spectrum (RamiGlyph 512-D)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.02,
        f"{dataset_name} ({split})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
        style="italic",
    )
    ax.set_box_aspect(1)
    ax.grid(True, linewidth=0.3, alpha=0.25, color="#aaaaaa")
    ax.set_axisbelow(True)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)
        ax.spines[spine].set_color("#444444")
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9,
        length=3,
        width=0.6,
        color="#444444",
    )
    legend = ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="#CCCCCC",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        markerscale=1.3,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.6,
    )
    legend.get_frame().set_linewidth(0.5)
    fig.tight_layout()
    save_figure_svg(fig, save_path)


def save_coordinate_table(path, labels, coordinate_name, coordinates):
    """Save two-dimensional embedding coordinates for reproducibility."""
    table = pd.DataFrame(
        {
            "bootstrap_id": np.arange(len(labels), dtype=int),
            "condition": labels,
            f"{coordinate_name}_1": coordinates[:, 0],
            f"{coordinate_name}_2": coordinates[:, 1],
        }
    )
    table.to_csv(path, index=False)


def visualize_bootstrap_spectrum(
    bootstrap_frame,
    save_dir,
    dataset_name,
    split,
    rand_seed=42,
    pca_dim=50,
    umap_neighbors=50,
):
    print("\nGenerating Bootstrap morphological spectrum...")
    os.makedirs(save_dir, exist_ok=True)

    all_features = np.vstack(bootstrap_frame["features"].values).astype(np.float32)
    all_labels = bootstrap_frame["condition"].to_numpy(dtype=str)
    if not np.isfinite(all_features).all():
        raise ValueError("Bootstrap features contain NaN or Inf")

    actual_pca_dim = min(pca_dim, len(all_features) - 1, all_features.shape[1])
    if actual_pca_dim < 2:
        raise ValueError("At least three Bootstrap vectors are required for plotting")

    print(f"  Computing PCA ({actual_pca_dim} dimensions)...")
    pca = PCA(n_components=actual_pca_dim, random_state=rand_seed)
    pca_coordinates = pca.fit_transform(all_features)
    file_prefix = f"{dataset_name}_{split}"

    pca_table = pd.DataFrame(
        pca_coordinates,
        columns=[f"PC{index + 1:02d}" for index in range(actual_pca_dim)],
    )
    pca_table.insert(0, "condition", all_labels)
    pca_table.insert(0, "bootstrap_id", np.arange(len(all_labels), dtype=int))
    pca_table.to_csv(
        os.path.join(save_dir, f"{file_prefix}_bootstrap_pca_coordinates.csv"),
        index=False,
    )
    pd.DataFrame(
        {
            "component": np.arange(1, actual_pca_dim + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(
        os.path.join(save_dir, f"{file_prefix}_pca_explained_variance.csv"),
        index=False,
    )

    print("  Computing t-SNE...")
    tsne = TSNE(
        n_components=2,
        random_state=rand_seed,
        perplexity=min(50, len(pca_coordinates) - 1),
    )
    tsne_coordinates = tsne.fit_transform(pca_coordinates)
    tsne_path = os.path.join(save_dir, f"{file_prefix}_bootstrap_tsne.svg")
    draw_embedding(
        tsne_coordinates,
        all_labels,
        dataset_name,
        split,
        "t-SNE",
        tsne_path,
    )
    save_coordinate_table(
        os.path.join(save_dir, f"{file_prefix}_bootstrap_tsne_coordinates.csv"),
        all_labels,
        "tSNE",
        tsne_coordinates,
    )
    print(f"  t-SNE plot saved to: {tsne_path}")

    print("  Computing UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        random_state=rand_seed,
        n_neighbors=min(umap_neighbors, len(pca_coordinates) - 1),
        min_dist=1.0,
        spread=3.0,
        metric="cosine",
    )
    umap_coordinates = reducer.fit_transform(pca_coordinates)
    umap_path = os.path.join(save_dir, f"{file_prefix}_bootstrap_umap.svg")
    draw_embedding(
        umap_coordinates,
        all_labels,
        dataset_name,
        split,
        "UMAP",
        umap_path,
    )
    save_coordinate_table(
        os.path.join(save_dir, f"{file_prefix}_bootstrap_umap_coordinates.csv"),
        all_labels,
        "UMAP",
        umap_coordinates,
    )
    print(f"  UMAP plot saved to: {umap_path}")


def save_bootstrap_summary(bootstrap_frame, save_dir, dataset_name, split):
    summary_path = os.path.join(
        save_dir, f"{dataset_name}_{split}_bootstrap_summary.txt"
    )

    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Bootstrap Analysis Summary\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Bootstrap Sample Counts:\n")
        f.write("-" * 40 + "\n")
        for condition in sorted(bootstrap_frame["condition"].unique()):
            count = len(bootstrap_frame[bootstrap_frame["condition"] == condition])
            f.write(f"  {condition}: {count} bootstrap samples\n")

        f.write("\n")

        f.write("Original Sample Counts (estimated):\n")
        f.write("-" * 40 + "\n")
        for condition in sorted(bootstrap_frame["condition"].unique()):
            indices_list = bootstrap_frame[bootstrap_frame["condition"] == condition][
                "bootstrap_indices"
            ].values
            all_indices = set()
            for indices in indices_list:
                all_indices.update(indices)
            f.write(f"  {condition}: ~{len(all_indices)} original samples\n")

        f.write("\n")

        if "feature_norm" not in bootstrap_frame.columns:
            bootstrap_frame["feature_norm"] = bootstrap_frame["features"].apply(
                lambda x: np.linalg.norm(x)
            )

        f.write("Feature Norm Statistics:\n")
        f.write("-" * 40 + "\n")
        for condition in sorted(bootstrap_frame["condition"].unique()):
            norms = bootstrap_frame[bootstrap_frame["condition"] == condition][
                "feature_norm"
            ].values
            f.write(f"  {condition}:\n")
            f.write(f"    Mean: {norms.mean():.4f}\n")
            f.write(f"    Std:  {norms.std():.4f}\n")
            f.write(f"    Min:  {norms.min():.4f}\n")
            f.write(f"    Max:  {norms.max():.4f}\n")

    print(f"\nBootstrap summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap analysis for RamiGlyph representations"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split to analyze",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name used in output files; defaults to the data-root folder name",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT, help="Model checkpoint path"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Configuration file path"
    )
    parser.add_argument("--data_root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Feature extraction batch size"
    )
    parser.add_argument(
        "--output_dir",
        default="./bootstrap_results",
        help="Output directory",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device"
    )
    parser.add_argument(
        "--N_bags", type=int, default=50, help="Number of bootstrap bags"
    )
    parser.add_argument("--n_samples", type=int, default=15, help="Samples per bag")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0,
        help="Sampling ratio; overrides a fixed sample count when positive",
    )
    parser.add_argument(
        "--replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample with replacement",
    )
    parser.add_argument("--rand_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pca_dim",
        "--pca-dim",
        dest="pca_dim",
        type=int,
        default=50,
        help="Maximum PCA dimensions used before t-SNE and UMAP",
    )
    parser.add_argument(
        "--umap_neighbors",
        "--umap-neighbors",
        dest="umap_neighbors",
        type=int,
        default=50,
        help="Maximum number of UMAP neighbors",
    )
    args = parser.parse_args()

    if args.pca_dim < 2:
        raise ValueError("PCA dimensions must be at least 2")
    if args.umap_neighbors < 2:
        raise ValueError("UMAP neighbors must be at least 2")

    apply_plot_style()
    set_seed(args.rand_seed)
    print(f"Random seed fixed to {args.rand_seed} for reproducibility.")

    dataset_name = args.dataset or os.path.basename(
        os.path.abspath(os.path.expanduser(args.data_root))
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    topo_config = config.get("topological", {})
    model = load_model(config, args.checkpoint, device)

    print("\n" + "=" * 60)
    print("Bootstrap Analysis Configuration")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {args.split}")
    print(f"N_bags: {args.N_bags}")
    print(f"n_samples: {args.n_samples}")
    print(f"Replacement: {args.replacement}")
    print(f"Ratio: {args.ratio}")
    print(f"PCA dimensions: {args.pca_dim}")
    print(f"UMAP neighbors: {args.umap_neighbors}")
    print("=" * 60 + "\n")

    print(f"Loading {args.split} dataset from {args.data_root}...")
    filtered_data, labels = load_labeled_graphs(
        MicrogliaDataset, args.data_root, args.split
    )
    print(f"Loaded samples: {len(filtered_data)}")
    print_label_distribution(labels, args.split)

    print("\nExtracting features...")
    features, valid_indices = extract_features(
        model, filtered_data, topo_config, device, args.batch_size
    )
    labels_valid = [labels[i] for i in valid_indices]
    print(f"Feature shape: {features.shape}")

    print("\nGrouping features by condition...")
    features_dict = {}
    for label in set(labels_valid):
        mask = np.array(labels_valid) == label
        features_dict[label] = features[mask]
        print(f"  {label}: {features_dict[label].shape[0]} samples")

    print("\n" + "=" * 60)
    print("Performing Bootstrap Resampling")
    print("=" * 60)

    bootstrap_frame = get_bootstrap_frame(
        features_dict,
        N_bags=args.N_bags,
        n_samples=args.n_samples,
        replacement=args.replacement,
        ratio=args.ratio if args.ratio > 0 else None,
        rand_seed=args.rand_seed,
        feature_type="array",
    )

    print(f"\nBootstrap frame shape: {bootstrap_frame.shape}")
    print(f"Total bootstrap samples: {len(bootstrap_frame)}")

    save_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    visualize_bootstrap_spectrum(
        bootstrap_frame,
        save_dir,
        dataset_name,
        args.split,
        rand_seed=args.rand_seed,
        pca_dim=args.pca_dim,
        umap_neighbors=args.umap_neighbors,
    )
    save_bootstrap_summary(bootstrap_frame, save_dir, dataset_name, args.split)

    bootstrap_csv_path = os.path.join(
        save_dir, f"{dataset_name}_{args.split}_bootstrap_frame.csv"
    )
    bootstrap_frame_to_save = bootstrap_frame.copy()
    bootstrap_frame_to_save["features"] = bootstrap_frame_to_save["features"].apply(
        lambda values: ",".join(map(str, values))
    )
    bootstrap_frame_to_save["bootstrap_indices"] = bootstrap_frame_to_save[
        "bootstrap_indices"
    ].apply(lambda indices: ",".join(map(str, indices)))
    bootstrap_frame_to_save.to_csv(bootstrap_csv_path, index=False)
    print(f"\nBootstrap frame saved to: {bootstrap_csv_path}")

    print("\n" + "=" * 60)
    print("Bootstrap Analysis Complete!")
    print("=" * 60)
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
