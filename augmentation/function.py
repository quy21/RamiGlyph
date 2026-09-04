import numpy as np


def neighbors_to_adjacency_attr(neighbors, node_type):
    """Build a dense adjacency matrix containing node-type edge codes."""
    n_nodes = len(neighbors)
    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    if hasattr(node_type, 'cpu'):
        node_type = node_type.cpu().numpy()

    for node_id, neighbor_list in neighbors.items():
        for neighbor_id in neighbor_list:
            type_idx_1 = int(np.argmax(node_type[node_id])) if len(node_type[node_id]) > 0 else 0
            type_idx_2 = int(np.argmax(node_type[neighbor_id])) if len(node_type[neighbor_id]) > 0 else 0
            edge_type = type_idx_1 * 5 + type_idx_2 + 1
            edge_type = min(edge_type, 31)
            adj[node_id, neighbor_id] = edge_type

    return adj


def neighbors_to_adjacency_pos(neighbors, pos):
    """Build a dense adjacency matrix containing Euclidean edge lengths."""
    n_nodes = len(neighbors)
    adj = np.zeros((n_nodes, n_nodes), dtype=float)

    if hasattr(pos, 'cpu'):
        pos = pos.cpu().numpy()

    for node_id, neighbor_list in neighbors.items():
        for neighbor_id in neighbor_list:
            distance = np.linalg.norm(pos[node_id] - pos[neighbor_id])
            adj[node_id, neighbor_id] = distance

    return adj


def adjacency_to_neighbors(adj_matrix):
    """Convert a dense or sparse adjacency matrix to neighbor sets."""

    if hasattr(adj_matrix, "toarray"):
        adj_matrix = adj_matrix.toarray()
    elif hasattr(adj_matrix, "cpu"):
        adj_matrix = adj_matrix.cpu().numpy()

    neighbors = {i: set() for i in range(adj_matrix.shape[0])}

    rows, cols = np.where(adj_matrix > 0)
    for row, col in zip(rows, cols):
        neighbors[row].add(col)

    return neighbors
