#!/usr/bin/env python

import math
import torch
import networkx as nx
import scipy.sparse as sp
import numpy as np
from typing import Sequence, Union, Tuple
from torch_geometric.utils import to_scipy_sparse_matrix

from augmentation.function import neighbors_to_adjacency_attr, neighbors_to_adjacency_pos, adjacency_to_neighbors


def compute_branch_level_from_neighbors(neighbors, soma_id=0):
    num_nodes = len(neighbors)
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
    

    level_array = torch.zeros(num_nodes, dtype=torch.long)
    for node_id, level in levels.items():
        level_array[node_id] = level
    
    return level_array


def compute_branch_id_from_neighbors(neighbors, soma_id=0):
    num_nodes = len(neighbors)
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
    

    branch_id_array = torch.zeros(num_nodes, dtype=torch.long)
    for node_id, bid in branch_ids.items():
        branch_id_array[node_id] = bid
    
    return branch_id_array


def remap_neighbors(x, soma_id=0):

    sorted_keys = sorted(x.keys())
    
    if soma_id in sorted_keys:

        sorted_keys.remove(soma_id)
        sorted_keys = [soma_id] + sorted_keys
    
    subsampled2new = {k: i for i, k in enumerate(sorted_keys)}

    # Re-map indices to 0..N-1
    ordered_x = {subsampled2new[k]: x[k] for k in sorted_keys}

    # Re-map keys of neighbors
    for k in ordered_x:
        ordered_x[k] = {subsampled2new[n] for n in ordered_x[k]}

    return ordered_x, subsampled2new

def subsample_graph(neighbors=None, not_deleted=None, keep_nodes=200, protected=[0], soma_id=0, generator=None):
    """
    Subsample graph.

    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
        keep_nodes: number of nodes to keep in graph
        protected: nodes to be excluded from subsampling
        soma_id: soma node ID (default 0)
        generator: torch.Generator for reproducibility
    """
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError('neighbors must be provided')


    protected_set = set(protected)
    

    if soma_id in neighbors:
        soma_neighbors = neighbors[soma_id]
        protected_set.update(soma_neighbors)
    

    protected = protected_set

    # indices as set in random order
    if generator is not None:
        perm = torch.randperm(k_nodes, generator=generator).tolist()
    else:
        perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    deleted = set()

    while len(deleted) < k_nodes - keep_nodes:

        while True:
            if len(all_indices) == 0:
                assert len(not_deleted) > keep_nodes, len(not_deleted)
                remaining = list(not_deleted - deleted)
                if generator is not None:
                    perm = torch.randperm(len(remaining), generator=generator).tolist()
                else:
                    perm = torch.randperm(len(remaining)).tolist()
                all_indices = np.array(remaining)[perm].tolist()

            idx = all_indices.pop()

            if idx not in deleted and len(neighbors[idx]) < 3 and idx not in protected:
                break

        if len(neighbors[idx]) == 2:
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)

        del neighbors[idx]
        deleted.add(idx)

    not_deleted = list(not_deleted - deleted)
    return neighbors, not_deleted


def get_leaf_branch_nodes(neighbors):
    """"
    Create list of candidates for leaf and branching nodes.
    Args:
        neighbors: dict of neighbors per node
    """
    all_nodes = list(neighbors.keys())
    leafs = [i for i in all_nodes if len(neighbors[i]) == 1]

    candidates = leafs
    next_nodes = []
    for l in leafs:
        next_nodes += [n for n in neighbors[l] if len(neighbors[n]) == 2]

    while next_nodes:
        s = next_nodes.pop(0)
        candidates.append(s)
        next_nodes += [n for n in neighbors[s] if
                       len(neighbors[n]) == 2 and n not in candidates and n not in next_nodes]

    return candidates


def compute_node_distances(idx, neighbors):
    """"
    Computation of node degree.
    Args:
        idx: index of node
        neighbors: dict of neighbors per node
    """
    queue = []
    queue.append(idx)

    degree = dict()
    degree[idx] = 0

    while queue:
        s = queue.pop(0)
        prev_dist = degree[s]

        for neighbor in neighbors[s]:
              if neighbor not in degree:
                queue.append(neighbor)
                degree[neighbor] = prev_dist + 1
    return degree


def drop_random_branch(nodes, neighbors, distances, keep_nodes=200, generator=None):
    """
    Removes a terminal branch. Starting nodes should be between
    branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: List of nodes of the graph
        neighbors: Dict of neighbors per node
        distances: Dict of distances of nodes to origin
        keep_nodes: Number of nodes to keep in graph
        generator: torch.Generator for reproducibility
    """
    if generator is not None:
        start = list(nodes)[torch.randint(len(nodes), (1,), generator=generator).item()]
    else:
        start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = list(neighbors[start])[0]

    # print(len(nodes))
    # print('node', nodes)
    #
    # print('start', start)
    # print('to', to)
    # print(len(distances))
    # print('distance[start]', distances[start])
    # print('distance[to]', distances[to])

    if distances[start] > distances[to]:
        start, to = to, start
    #
    # print('len node', len(nodes))
    # print('len neighbor', len(neighbors))
    # print('len distance',len(distances))

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0)
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes]

    if len(neighbors) - len(drop_nodes) < keep_nodes:
        return neighbors, set()
    else:
        # Delete nodes.
        for key in drop_nodes:
            if key in neighbors:
                for k in neighbors[key]:
                    neighbors[k].remove(key)
                del neighbors[key]

        return neighbors, set(drop_nodes)


class RanDomReduceNodes:
    def __init__(self, keep_node: Union[int], n_branch: int, soma_id: int = 0, seed=None):
        self.keep_node = keep_node
        self.n_branch = n_branch
        self.soma_id = soma_id
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, data):
        # print(data)

        # for i in range(batch_size):
        #
        adj = to_scipy_sparse_matrix(data.edge_index)

        # G = nx.Graph(adj)
        #
        # print('aaaaa')

        neighbors = adjacency_to_neighbors(adj_matrix=adj)
        # print('neighbor', len(neighbors))

        leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
        # print('leaf branch nodes', leaf_branch_nodes)
        # Using the distances we can infer the direction of an edge.

        # distances = compute_node_distances(self.soma_id, neighbors)
        distances = data.distance
        # print('distance', len(distances))
        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors)))

        for i in range(self.n_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes,
                                                       neighbors,
                                                       distances,
                                                       keep_nodes=self.keep_node,
                                                       generator=self.generator)

            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes

            if len(leaf_branch_nodes) == 0:
                break


        neighbors, not_deleted = subsample_graph(neighbors=neighbors,
                                                 not_deleted=not_deleted,
                                                 keep_nodes=self.keep_node,
                                                 protected=[0],
                                                 soma_id=self.soma_id,
                                                 generator=self.generator)
        neighbors, subsampled2new = remap_neighbors(neighbors)

        # print(subsampled2new)


        features = data.x[list(subsampled2new.keys())]
        pos = data.pos[list(subsampled2new.keys())]
        node_types = features[:, 4:]

        subsampling_adj = neighbors_to_adjacency_pos(neighbors, pos)
        edge_index_temp = sp.coo_matrix(subsampling_adj)
        rows, cols = edge_index_temp.row, edge_index_temp.col
        edge_index = torch.LongTensor(np.vstack((rows, cols)))
        edge_weight = torch.Tensor(edge_index_temp.data)

        subsampling_adj_atr = neighbors_to_adjacency_attr(neighbors, node_types)
        attr_value = subsampling_adj_atr[rows, cols]
        edge_attr = torch.LongTensor(attr_value)


        data.x = features
        data.edge_index = edge_index
        data.pos = pos
        data.edge_attr = edge_attr
        data.edge_weight = edge_weight


        data.branch_level = compute_branch_level_from_neighbors(neighbors, soma_id=0)
        

        data.branch_id = compute_branch_id_from_neighbors(neighbors, soma_id=0)
        

        if hasattr(data, 'distance') and data.distance is not None:

            new_distances = compute_node_distances(0, neighbors)
            dist_array = torch.zeros(len(neighbors))
            for i in range(len(neighbors)):
                dist_array[i] = new_distances[i]
            data.distance = dist_array
            
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.keep_node})'
