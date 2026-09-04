"""Dataset helpers shared by RamiGlyph evaluation scripts."""

from collections import Counter
from pathlib import Path


def build_folder_label_index(data_root, split):
    """Map each SWC stem to its first-level category folder."""
    split_dir = Path(data_root).expanduser().resolve() / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

    label_index = {}
    ambiguous_ids = set()

    category_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not category_dirs:
        raise ValueError(
            f"No category folders found in {split_dir}. "
            "Expected '<split>/<label>/*.swc'."
        )

    for category_dir in category_dirs:
        label = category_dir.name
        for swc_path in category_dir.rglob("*.swc"):
            cell_id = swc_path.stem
            existing_label = label_index.get(cell_id)
            if existing_label is not None and existing_label != label:
                ambiguous_ids.add(cell_id)
            else:
                label_index[cell_id] = label

    if not label_index:
        raise ValueError(f"No SWC files found in category folders under {split_dir}")

    return label_index, ambiguous_ids


def get_graph_label(graph, label_index, ambiguous_ids):
    """Return a graph label stored during processing or inferred by cell ID."""
    graph_label = getattr(graph, "group_label", None)
    if graph_label is not None and str(graph_label):
        return str(graph_label)

    cell_id = str(getattr(graph, "cell_id", ""))
    if not cell_id:
        raise ValueError("A graph is missing the required 'cell_id' attribute")
    if cell_id in ambiguous_ids:
        raise ValueError(
            f"Cell ID '{cell_id}' occurs in multiple category folders. "
            "Reprocess the dataset so each graph stores its folder label, or use "
            "unique SWC file names."
        )

    label = label_index.get(cell_id)
    if label is None:
        raise ValueError(
            f"No category folder could be resolved for graph '{cell_id}' in the raw data"
        )
    return label


def load_labeled_graphs(dataset_class, data_root, split):
    """Load a graph split and derive labels from first-level folders."""
    dataset = dataset_class(root=data_root, split=split, transform=None)
    label_index, ambiguous_ids = build_folder_label_index(data_root, split)

    graphs = []
    labels = []
    for graph in dataset:
        graphs.append(graph)
        labels.append(get_graph_label(graph, label_index, ambiguous_ids))

    if not graphs:
        raise ValueError(f"No valid graphs found in the '{split}' split")

    return graphs, labels


def print_label_distribution(labels, split):
    """Print category counts for one dataset split."""
    print(f"\n{split.capitalize()} label distribution:")
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")
