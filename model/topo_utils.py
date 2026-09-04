"""Topological feature extraction for morphology graphs."""

import gudhi as gd
import networkx as nx
import numpy as np


def apply_graph_extended_persistence_0d(A, filtration_val):
    """Compute the zero-dimensional extended persistence diagram."""
    adjacency = A
    filtration_values = filtration_val
    num_vertices = adjacency.shape[0]
    row_indices, column_indices = np.where(np.triu(adjacency))
    simplex_tree = gd.SimplexTree()

    for vertex_id in range(num_vertices):
        simplex_tree.insert([vertex_id], filtration=-1e10)
    for row, column in zip(row_indices, column_indices):
        simplex_tree.insert([row, column], filtration=-1e10)

    for vertex_id in range(num_vertices):
        simplex_tree.assign_filtration([vertex_id], filtration_values[vertex_id])

    simplex_tree.make_filtration_non_decreasing()
    simplex_tree.extend_filtration()
    persistence = simplex_tree.extended_persistence()

    ordinary_zero, extended_zero = persistence[0], persistence[2]
    ordinary_diagram = _extract_zero_dimensional_pairs(ordinary_zero)
    extended_diagram = _extract_zero_dimensional_pairs(extended_zero)
    return np.concatenate([ordinary_diagram, extended_diagram], axis=0)


def _extract_zero_dimensional_pairs(persistence):
    """Return ordered birth-death pairs for dimension zero."""
    if not persistence:
        return np.empty([0, 2])

    return np.vstack(
        [
            np.array(
                [
                    [
                        min(point[1][0], point[1][1]),
                        max(point[1][0], point[1][1]),
                    ]
                ]
            )
            for point in persistence
            if point[0] == 0
        ]
    )


def persistence_images(
    dgm,
    resolution=[50, 50],
    normalization=True,
    bandwidth=1.0,
    power=1.0,
):
    """Convert a persistence diagram into a persistence image."""
    diagram = dgm
    births, deaths = diagram[:, 0], diagram[:, 1]

    x_min, x_max = births.min(), births.max()
    if x_min == x_max:
        x_min, x_max = x_min - 0.1, x_max + 0.1

    y_min, y_max = deaths.min(), deaths.max()
    if y_min == y_max:
        y_min, y_max = y_min - 0.1, y_max + 0.1

    x_axis = np.linspace(x_min, x_max, resolution[0])
    y_axis = np.linspace(y_min, y_max, resolution[1])
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    grid_x = grid_x[:, :, np.newaxis]
    grid_y = grid_y[:, :, np.newaxis]

    birth_points = np.reshape(diagram[:, 0], [1, 1, -1])
    death_points = np.reshape(diagram[:, 1], [1, 1, -1])
    weights = np.abs(death_points - birth_points) ** power
    point_distances = np.sqrt(
        (grid_x - birth_points) ** 2 + (grid_y - death_points) ** 2
    )

    image = np.multiply(
        weights,
        np.exp(-(point_distances**2) / bandwidth),
    ).sum(axis=2)

    if not normalization:
        return image

    min_value, max_value = np.min(image), np.max(image)
    if max_value > min_value:
        return (image - min_value) / (max_value - min_value)
    return np.zeros(resolution) + 1e-6


def compute_filtration_from_soma(data, soma_node_id=0):
    """Compute graph-distance filtration values from the soma node."""
    edge_index = data.edge_index.numpy().T
    num_nodes = data.x.size(0) if hasattr(data, "x") else data.num_nodes

    try:
        graph = nx.from_edgelist(edge_index)
        if soma_node_id not in graph.nodes():
            soma_node_id = 0

        distances = nx.single_source_shortest_path_length(
            graph,
            source=soma_node_id,
        )
        filtration_values = np.array(
            [distances.get(node_id, 0) for node_id in range(num_nodes)]
        )
    except Exception as error:
        print(f"Warning: Failed to compute soma distance: {error}")
        try:
            graph = nx.from_edgelist(edge_index)
            betweenness = nx.betweenness_centrality(graph)
            filtration_values = np.array(
                [betweenness.get(node_id, 0) for node_id in range(num_nodes)]
            )
        except Exception as fallback_error:
            print(
                "Warning: Failed to compute betweenness centrality: "
                f"{fallback_error}"
            )
            filtration_values = np.zeros(num_nodes)
            for source, target in edge_index:
                filtration_values[source] += 1
                filtration_values[target] += 1

    return filtration_values


def compute_topological_features_single(
    data,
    resolution=50,
    soma_node_id=0,
):
    """Compute one square persistence image for a morphology graph."""
    edge_index = data.edge_index.numpy().T
    graph = nx.from_edgelist(edge_index)
    adjacency = nx.adjacency_matrix(graph).toarray()
    filtration_values = compute_filtration_from_soma(data, soma_node_id)
    diagram = apply_graph_extended_persistence_0d(
        adjacency,
        filtration_values,
    )
    return persistence_images(
        diagram,
        resolution=[resolution, resolution],
    )
