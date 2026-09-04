import os
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


from dataloader.swc_utils import subsample_graph, remap_neighbors
from augmentation.reduce_node import (
    compute_node_distances,
    compute_branch_level_from_neighbors,
    compute_branch_id_from_neighbors,
)
from augmentation.function import (
    neighbors_to_adjacency_attr,
    neighbors_to_adjacency_pos,
)


def parse_swc_file(swc_path):
    data = []
    with open(swc_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    data.append([float(x) for x in parts])
                except ValueError:
                    continue

    if len(data) == 0:
        return None, None

    data = np.array(data)
    n_nodes = len(data)

    idx_map = {int(data[i, 0]): i for i in range(n_nodes)}

    coords = data[:, 2:5]  # X, Y, Z
    radius = data[:, 5:6]  # Radius
    types = data[:, 1].astype(int)  # Type

    type_one_hot = np.zeros((n_nodes, 5))
    for i, t in enumerate(types):
        if 1 <= t <= 5:
            type_one_hot[i, t - 1] = 1
        else:
            type_one_hot[i, 0] = 1

    features = np.concatenate([coords, radius, type_one_hot], axis=1)

    neighbors = {i: set() for i in range(n_nodes)}
    for i in range(n_nodes):
        parent_idx = int(data[i, 6])
        if parent_idx != -1 and parent_idx in idx_map:
            parent_i = idx_map[parent_idx]
            neighbors[i].add(parent_i)
            neighbors[parent_i].add(i)

    return features, neighbors


class MicrogliaDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        assert split in ["train", "val", "test", "all"]
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.subset = subset

        path = os.path.join(self.processed_dir, f"{split}.pt")
        if os.path.exists(path):
            self.data, self.slices = torch.load(path, weights_only=False)
        else:
            self.process()
            self.data, self.slices = torch.load(path, weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["train.pt", "val.pt", "all.pt"]

    def download(self):
        pass

    def process(self):
        root_path = Path(self.root)

        train_dir = root_path / "train"
        val_dir = root_path / "val"

        if not all([train_dir.exists(), val_dir.exists()]):
            raise ValueError(
                f"Dataset directories missing in {self.root}. Required: train, val"
            )

        def collect_swc_files(split_dir):
            swc_files = []
            for swc_file in Path(split_dir).rglob("*.swc"):
                swc_files.append(swc_file)
            return sorted(swc_files)

        splits_files = {
            "train": collect_swc_files(train_dir),
            "val": collect_swc_files(val_dir),
        }
        splits_files["all"] = splits_files["train"] + splits_files["val"]

        print("\n" + "=" * 60)
        print("Dataset Statistics:")
        print("=" * 60)
        for split_name, files in splits_files.items():
            if split_name != "all":
                print(f"{split_name:10s}: {len(files):4d} samples")
        print(f'{"all":10s}: {len(splits_files["all"]):4d} samples')
        print("=" * 60 + "\n")

        for split in ["train", "val", "all"]:
            cell_ids = splits_files[split]

            pbar = tqdm(total=len(cell_ids))
            pbar.set_description(f"Processing {split} dataset")

            data_list = []
            for cell_id in cell_ids:
                if cell_id.name == "desktop.ini":
                    continue

                relative_path = cell_id.relative_to(root_path)
                group_label = (
                    relative_path.parts[1] if len(relative_path.parts) > 2 else ""
                )

                soma_id = 0
                features, neighbors = parse_swc_file(cell_id)

                if features is None or neighbors is None:
                    pbar.update(1)
                    continue

                neighbors, not_deleted = subsample_graph(
                    neighbors=neighbors,
                    not_deleted=set(range(len(neighbors))),
                    keep_nodes=1000,
                    protected=[soma_id],
                )

                neighbors, subsampled2new = remap_neighbors(neighbors)

                features = features[sorted(subsampled2new.keys())]

                node_distances = compute_node_distances(0, neighbors)
                a = np.zeros(len(node_distances))
                for i in range(len(node_distances)):
                    a[i] = node_distances[i]
                distances = torch.Tensor(a)

                branch_level = compute_branch_level_from_neighbors(neighbors, soma_id=0)
                branch_id = compute_branch_id_from_neighbors(neighbors, soma_id=0)

                x = torch.Tensor(features)
                pos = features[:, :3]
                node_types = features[:, 4:]

                pos = (pos - np.min(pos)) / (np.max(pos) - np.min(pos))
                pos = torch.Tensor(pos)

                adj_attr = neighbors_to_adjacency_attr(neighbors, node_type=node_types)

                G_struct = nx.Graph()
                G_struct.add_nodes_from(neighbors.keys())
                for u, nbrs in neighbors.items():
                    for v in nbrs:
                        if u < v:
                            G_struct.add_edge(u, v)

                if nx.number_connected_components(G_struct) > 1:
                    print("attr")

                edge_index_temp_attr = sp.coo_matrix(adj_attr)
                values_attr = edge_index_temp_attr.data
                edge_attr = torch.LongTensor(values_attr)

                indices_attr = np.vstack(
                    (edge_index_temp_attr.row, edge_index_temp_attr.col)
                )
                edge_index_attr = torch.LongTensor(indices_attr)

                adj_pos = neighbors_to_adjacency_pos(neighbors, pos=pos)
                edge_index_temp_pos = sp.coo_matrix(adj_pos)
                values_pos = edge_index_temp_pos.data
                edge_weight = torch.Tensor(values_pos)

                if edge_index_attr.shape[1] == 0:
                    pbar.update(1)
                    continue

                data = Data(
                    x=x,
                    edge_index=edge_index_attr,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                    pos=pos,
                    distance=distances,
                    branch_level=branch_level,
                    branch_id=branch_id,
                    cell_id=cell_id.stem,
                    group_label=group_label,
                    source_path=relative_path.as_posix(),
                )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()
            torch.save(
                self.collate(data_list), os.path.join(self.processed_dir, f"{split}.pt")
            )
