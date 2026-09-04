#!/usr/bin/env python

import scipy.sparse as sp
import math
import numbers
import random
from itertools import repeat
from typing import Sequence, Union, Tuple
from augmentation.function import neighbors_to_adjacency_pos, adjacency_to_neighbors

import numpy as np
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from scipy.spatial.transform import Rotation as R


def rotate_graph(pos_matrix, rng=None):
    ''' Randomly rotate graph in xyz-direction.

    Args:
        pos_matrix: Matrix with xyz-node positions (N x 3).
        axis: Axis around which to rotate. Defaults to `None`,
            in which case no rotation is performed.
        rng: Random number generator for reproducibility.
    '''
    if rng is None:
        rng = random.Random()

    axis = rng.choice([0, 1, 2])

    rotation_matrix = R.random(random_state=rng.randint(0, 2**31-1)).as_matrix()

    if axis == 0: # x
        rotation_matrix[0, 1] = 0
        rotation_matrix[0, 2] = 0
        rotation_matrix[0, 0] = 1
        rotation_matrix[1, 0] = 0
        rotation_matrix[2, 0] = 0
    elif axis == 1:  # y
        rotation_matrix[0, 1] = 0
        rotation_matrix[1, 0] = 0
        rotation_matrix[1, 1] = 1
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 1] = 0
    elif axis == 2:  # z
        rotation_matrix[0, 2] = 0
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 2] = 1
        rotation_matrix[2, 0] = 0
        rotation_matrix[2, 1] = 0

    # Convert rotation matrix to torch tensor
    is_tensor = hasattr(pos_matrix, 'cpu')
    if is_tensor:
        # Work entirely in PyTorch
        rotation_tensor = torch.tensor(rotation_matrix, dtype=pos_matrix.dtype, device=pos_matrix.device)
        rot_pos_matrix = torch.matmul(pos_matrix, rotation_tensor)
    else:
        # Work in numpy
        pos_np = np.array(pos_matrix, dtype=np.float64)
        rotation_matrix = np.array(rotation_matrix, dtype=np.float64)
        rot_pos_matrix = np.matmul(pos_np, rotation_matrix)
    
    return rot_pos_matrix


def random_flip(pos, axis=None, prob=0.5, rng=None):
    if rng is None:
        rng = random.Random()
    
    if rng.random() < prob:
        flip_pos = pos.clone()
        if axis is None:
            axis = rng.choice([0, 1, 2])
        flip_pos[..., axis] = -flip_pos[..., axis]
        return flip_pos
    return pos


def jitter_node_pos(node_positions, scale=0.1, generator=None):
    """
    Randomly jitter nodes in xyz-direction.

    Args:
        node_positions: Matrix with xyz-node positions (N x 3).
        scale: Scale factor for jittering.
        generator: torch.Generator for reproducibility.
    """
    (n, dim), t = node_positions.size(), scale
    if isinstance(t, numbers.Number):
        t = list(repeat(t, times=n))
    assert len(t) == n
    pos = node_positions.clone()
    ts = []
    for d in range(n):
        if generator is not None:
            ts.append((torch.rand(dim, generator=generator) * 2 - 1) * abs(t[d]))
        else:
            ts.append(pos.new_empty(dim).uniform_(-abs(t[d]), abs(t[d])))

    for i in range(n):
        pos[i] = node_positions[i] + ts[i]
    return pos


# def jitter_node(node_positions, scale=0.1):
#
#     return node_positions + (torch.randn(*node_positions.shape).numpy() * scale)


def translate_soma_pos(node_positions, scale=1, generator=None):
    """
    Randomly translate the position of the entire grpah.

    Args:
        node_positions: Matrix with xyz-node positions (N x 3).
        scale: Scale factor for jittering.
        generator: torch.Generator for reproducibility.
    """
    new_node_features = node_positions.clone()
    if generator is not None:
        jitter = torch.randn(3, generator=generator).numpy() * scale
    else:
        jitter = torch.randn(3).numpy() * scale
    new_node_features[:, :3] += jitter
    return new_node_features


class Augment_node_position:
    def __init__(self, trans_scale, jitter_scale, rota=True, flip=True, flip_prob=0.5, seed=None):
        self.trans_scale = trans_scale
        self.scale = jitter_scale
        self.rota = rota
        self.flip = flip
        self.flip_prob = flip_prob
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, data):

        adj = to_scipy_sparse_matrix(data.edge_index)
        neighbors = adjacency_to_neighbors(adj_matrix=adj)

        features = data.x

        pos = features[:, :3]
        if self.rota:
            rot_pos = rotate_graph(pos, rng=self.rng)
        else:
            rot_pos = pos
        

        if self.flip:
            flip_pos = random_flip(rot_pos, axis=None, prob=self.flip_prob, rng=self.rng)
        else:
            flip_pos = rot_pos
        
        jitter_pos = jitter_node_pos(flip_pos, scale=self.scale, generator=self.generator)
        translate_pos = translate_soma_pos(jitter_pos, scale=self.trans_scale, generator=self.generator)
        # translate_pos = tensor.cpu().numpy()

        features[:, :3] = translate_pos


        translate_pos = (translate_pos - torch.min(translate_pos))/(torch.max(translate_pos) - torch.min(translate_pos))


        subsampling_adj = neighbors_to_adjacency_pos(neighbors, translate_pos)
        edge_index_temp = sp.coo_matrix(subsampling_adj)
        weight_values = edge_index_temp.data
        edge_weight = torch.Tensor(weight_values)


        data.weight = edge_weight

        data.pos = translate_pos

        data.x = features

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.degree})'
