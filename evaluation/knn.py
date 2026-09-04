#!/usr/bin/env python
"""Evaluate RamiGlyph representations with a K-nearest-neighbor classifier.

Organize SWC files by split and class label::

    MyDataset/
    ├── train/
    │   ├── Control/*.swc
    │   └── Treatment/*.swc
    └── val/
        ├── Control/*.swc
        └── Treatment/*.swc

The first-level folder below each split is used as the class label. The training
split provides the KNN reference features, and ``--split`` selects the query
split. Example::

    python evaluation/knn.py \
        --data_root /path/to/MyDataset \
        --split val \
        --checkpoint checkpoints/best_checkpoint.pt
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch_geometric.transforms as T
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dataloader.microglia import MicrogliaDataset
from evaluation.data_utils import load_labeled_graphs, print_label_distribution
from model.dual_branch_model import build_dual_branch_swav_model
from model.topo_utils import compute_topological_features_single


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config.json")
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints", "best_checkpoint.pt")


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


def knn_evaluate(
    train_features,
    train_labels,
    test_features,
    test_labels,
    k_values=(1, 3, 5, 10, 20),
    top_k_values=(1, 3, 5),
):
    le = LabelEncoder()
    le.fit(train_labels)

    unseen_labels = sorted(set(test_labels) - set(le.classes_))
    if unseen_labels:
        raise ValueError(
            "Evaluation labels are missing from the training split: "
            + ", ".join(unseen_labels)
        )

    train_labels_encoded = le.transform(train_labels)
    test_labels_encoded = le.transform(test_labels)
    class_indices = np.arange(len(le.classes_))

    results = {}

    for k in k_values:
        if k > len(train_features):
            print(
                f"Skipping k={k}: only {len(train_features)} training samples are available."
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"KNN Evaluation (k={k})")
        print("=" * 60)

        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(train_features, train_labels_encoded)

        predictions = knn.predict(test_features)
        probabilities = knn.predict_proba(test_features)

        overall_acc = accuracy_score(test_labels_encoded, predictions)
        balanced_acc = balanced_accuracy_score(test_labels_encoded, predictions)
        macro_f1 = f1_score(
            test_labels_encoded, predictions, labels=class_indices, average="macro"
        )

        cm = confusion_matrix(test_labels_encoded, predictions, labels=class_indices)
        per_class_acc = np.divide(
            cm.diagonal(),
            cm.sum(axis=1),
            out=np.zeros(len(class_indices), dtype=float),
            where=cm.sum(axis=1) != 0,
        )
        mean_acc = np.mean(per_class_acc)

        top_k_accs = {}
        for top_k in top_k_values:
            if top_k < len(le.classes_):
                if len(le.classes_) == 2 and top_k == 1:
                    top_k_acc = overall_acc
                else:
                    top_k_acc = top_k_accuracy_score(
                        test_labels_encoded,
                        probabilities,
                        k=top_k,
                        labels=class_indices,
                    )
                top_k_accs[f"top_{top_k}"] = top_k_acc

        report = classification_report(
            test_labels_encoded,
            predictions,
            labels=class_indices,
            target_names=le.classes_,
            zero_division=0,
        )
        results[f"k={k}"] = {
            "overall_accuracy": overall_acc,
            "mean_accuracy": mean_acc,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "per_class_accuracy": dict(zip(le.classes_, per_class_acc)),
            "confusion_matrix": cm,
            "top_k_accuracy": top_k_accs,
            "class_names": le.classes_.tolist(),
            "classification_report": report,
        }

        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Macro-F1: {macro_f1:.4f}")

        print("\nPer-class Accuracy:")
        for cls, acc in zip(le.classes_, per_class_acc):
            print(f"  {cls}: {acc:.4f}")

        print("\nTop-k Accuracy:")
        for k_name, acc in top_k_accs.items():
            print(f"  {k_name}: {acc:.4f}")

        print("\nClassification Report:")
        print(report)

    if not results:
        raise ValueError("None of the requested k values is valid for the training set")

    return results


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def save_results(
    results, save_dir, dataset_name, split, subfolder=None, file_prefix=None
):
    dataset_dir = os.path.join(save_dir, dataset_name)
    if subfolder is not None:
        dataset_dir = os.path.join(dataset_dir, subfolder)
    os.makedirs(dataset_dir, exist_ok=True)

    if file_prefix is None:
        file_prefix = dataset_name

    txt_path = os.path.join(dataset_dir, f"{file_prefix}_{split}_results.txt")
    with open(txt_path, "w") as f:
        f.write("KNN Evaluation Results\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Split: {split}\n")
        f.write("=" * 60 + "\n\n")

        for k_name, metrics in results.items():
            f.write(f"\n{k_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Macro-F1: {metrics['macro_f1']:.4f}\n")

            f.write("\nPer-class Accuracy:\n")
            for cls, acc in metrics["per_class_accuracy"].items():
                f.write(f"  {cls}: {acc:.4f}\n")

            f.write("\nTop-k Accuracy:\n")
            for k_name_inner, acc in metrics["top_k_accuracy"].items():
                f.write(f"  {k_name_inner}: {acc:.4f}\n")

            f.write("\nClassification Report:\n")
            f.write(metrics["classification_report"])
            f.write("\n")

            cm_path = os.path.join(
                dataset_dir, f"{file_prefix}_{split}_{k_name}_confusion_matrix.png"
            )
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                metrics["class_names"],
                cm_path,
                title=f"{dataset_name} - {split} - {k_name}",
            )

    print(f"\nResults saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="KNN evaluation for RamiGlyph representations"
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
        help="Dataset split to evaluate against the training reference set",
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
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="Values of k for KNN evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Feature extraction batch size"
    )
    parser.add_argument(
        "--output_dir", default="./knn_results", help="Output directory"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device"
    )
    args = parser.parse_args()

    set_seed(42)
    print("Random seed fixed to 42 for reproducibility.")

    dataset_name = args.dataset or os.path.basename(
        os.path.abspath(os.path.expanduser(args.data_root))
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    topo_config = config.get("topological", {})

    model = load_model(config, args.checkpoint, device)

    print(f"\nLoading train dataset from {args.data_root}...")
    train_data, train_labels = load_labeled_graphs(
        MicrogliaDataset, args.data_root, "train"
    )
    print(f"Loaded train samples: {len(train_data)}")
    print_label_distribution(train_labels, "train")

    print("\nExtracting train features...")
    train_features, train_valid_indices = extract_features(
        model, train_data, topo_config, device, args.batch_size
    )
    train_labels_valid = [train_labels[i] for i in train_valid_indices]
    print(f"Train feature shape: {train_features.shape}")

    print(f"\nLoading {args.split} dataset from {args.data_root}...")
    test_data, test_labels = load_labeled_graphs(
        MicrogliaDataset, args.data_root, args.split
    )
    print(f"Loaded {args.split} samples: {len(test_data)}")
    print_label_distribution(test_labels, args.split)

    print(f"\nExtracting {args.split} features...")
    test_features, test_valid_indices = extract_features(
        model, test_data, topo_config, device, args.batch_size
    )
    test_labels_valid = [test_labels[i] for i in test_valid_indices]
    print(f"{args.split.capitalize()} feature shape: {test_features.shape}")

    results = knn_evaluate(
        train_features,
        train_labels_valid,
        test_features,
        test_labels_valid,
        k_values=args.k_values,
        top_k_values=[1, 3, 5],
    )

    save_results(results, args.output_dir, dataset_name, args.split)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
