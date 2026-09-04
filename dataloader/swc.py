"""Read, inspect, modify, and write SWC morphology files."""

import copy


NODE_ID = "id"
NODE_TYPE = "type"
NODE_X = "x"
NODE_Y = "y"
NODE_Z = "z"
NODE_R = "radius"
NODE_PN = "parent"
SWC_COLUMNS = [NODE_ID, NODE_TYPE, NODE_X, NODE_Y, NODE_Z, NODE_R, NODE_PN]

NODE_TREE_ID = "tree_id"
NODE_CHILDREN = "children"


class Compartment(dict):
    """Dictionary representation of one SWC compartment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (
            NODE_ID not in self
            or NODE_TYPE not in self
            or NODE_X not in self
            or NODE_Y not in self
            or NODE_Z not in self
            or NODE_R not in self
            or NODE_PN not in self
        ):
            raise ValueError(
                "Compartment was not initialized with requisite fields"
            )
        self[NODE_TREE_ID] = -1
        self[NODE_CHILDREN] = []

    def print_node(self):
        """Print the compartment fields in SWC order."""
        print(
            "%d %d %.4f %.4f %.4f %.4f %d %s %d"
            % (
                self[NODE_ID],
                self[NODE_TYPE],
                self[NODE_X],
                self[NODE_Y],
                self[NODE_Z],
                self[NODE_R],
                self[NODE_PN],
                str(self[NODE_CHILDREN]),
                self[NODE_TREE_ID],
            )
        )


def read_swc(file_name):
    """Read an SWC file and return a reconstructed morphology."""
    compartments = []
    line_num = 1
    try:
        with open(file_name, "r") as swc_file:
            for line in swc_file:
                if line.lstrip().startswith("#"):
                    continue
                if line.isspace():
                    continue

                tokens = line.lstrip().split()
                compartment = Compartment(
                    {
                        NODE_ID: int(tokens[0]),
                        NODE_TYPE: int(tokens[1]),
                        NODE_X: float(tokens[2]),
                        NODE_Y: float(tokens[3]),
                        NODE_Z: float(tokens[4]),
                        NODE_R: float(tokens[5]),
                        NODE_PN: int(tokens[6].rstrip()),
                    }
                )
                compartments.append(compartment)
                line_num += 1

    except ValueError:
        error = "File not recognized as valid SWC file.\n"
        error += "Problem parsing line %d\n" % line_num
        if line is not None:
            error += "Content: '%s'\n" % line
        raise IOError(error)

    return Morphology(compartment_list=compartments)


class Morphology:
    """SWC compartments with tree, soma, and pruning helpers."""

    SOMA = 1
    AXON = 2
    DENDRITE = 3
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4

    NODE_TYPES = [SOMA, AXON, DENDRITE, BASAL_DENDRITE, APICAL_DENDRITE]

    def __init__(self, compartment_list=None, compartment_index=None):
        self._compartment_list = []
        self._compartment_index = {}
        self._tree_list = []

        if compartment_list:
            self.compartment_list = compartment_list
        elif compartment_index:
            self.compartment_index = compartment_index

        num_errors = self._check_consistency()
        if num_errors > 0:
            raise ValueError("Morphology appears to be inconsistent")

        self._soma = None
        for i in range(len(self.compartment_list)):
            segment = self.compartment_list[i]
            if segment[NODE_TYPE] == Morphology.SOMA and segment[NODE_PN] < 0:
                if self._soma is not None:
                    raise ValueError("Multiple somas detected in SWC file")
                self._soma = segment

    @property
    def compartment_list(self):
        """Return the ordered compartment list."""
        return self._compartment_list

    @compartment_list.setter
    def compartment_list(self, compartment_list):
        """Replace the compartments and rebuild internal relationships."""
        self._set_compartments(compartment_list)

    @property
    def num_trees(self):
        """Return the number of disconnected trees."""
        return len(self._tree_list)

    @property
    def num_nodes(self):
        """Return the number of compartments."""
        return len(self.compartment_list)

    @property
    def soma(self):
        """Return the root soma compartment, if present."""
        return self._soma

    @property
    def root(self):
        """Return the root soma compartment; retained for compatibility."""
        return self._soma

    def tree(self, tree_id):
        """Return a connected tree by index, or None if it is absent."""
        if tree_id < 0 or tree_id >= len(self._tree_list):
            return None
        return self._tree_list[tree_id]

    def node(self, node_id):
        return self._resolve_node_type(node_id)

    def parent_of(self, segment):
        return None

    def children_of(self, segment):
        return [
            self._compartment_list[child_id]
            for child_id in segment[NODE_CHILDREN]
        ]

    def _set_compartments(self, compartment_list):
        self._compartment_list = []
        for compartment in compartment_list:
            segment = copy.copy(compartment)
            segment[NODE_TREE_ID] = -1
            segment[NODE_CHILDREN] = []
            self._compartment_list.append(segment)
        self._reconstruct()

    def _reconstruct(self):
        """Compact node IDs and rebuild parent, child, index, and tree data."""
        remap = {}
        for i in range(len(self.compartment_list)):
            remap[i] = -1

        new_id = 0
        compacted = []
        for segment in self.compartment_list:
            if segment is not None:
                remap[segment[NODE_ID]] = new_id
                segment[NODE_ID] = new_id
                compacted.append(segment)
                new_id += 1

        for segment in compacted:
            if segment[NODE_PN] >= 0:
                segment[NODE_PN] = remap[segment[NODE_PN]]

        self._compartment_list = compacted
        for segment in self.compartment_list:
            segment[NODE_CHILDREN] = []
        for segment in self.compartment_list:
            parent_id = segment[NODE_PN]
            if parent_id >= 0:
                self.compartment_list[parent_id][NODE_CHILDREN].append(
                    segment[NODE_ID]
                )

        self._separate_trees()
        self._compartment_index = {
            compartment[NODE_ID]: compartment
            for compartment in self.compartment_list
        }

        for segment in self._compartment_list:
            segment[NODE_CHILDREN] = []
        for segment in self._compartment_list:
            if segment[NODE_PN] >= 0:
                self._compartment_list[segment[NODE_PN]][NODE_CHILDREN].append(
                    segment[NODE_ID]
                )

        for i in range(len(self.compartment_list)):
            if i != self.node(i)[NODE_ID]:
                raise RuntimeError(
                    "Internal error detected -- compartment list not properly formed"
                )

    def _separate_trees(self):
        """Build connected-tree lists and assign their identifiers."""
        trees = []
        for segment in self.compartment_list:
            segment[NODE_TREE_ID] = -1

        for segment in self.compartment_list:
            local_trees = []
            parent_id = segment[NODE_PN]
            if (
                parent_id >= 0
                and self.compartment_list[parent_id][NODE_TREE_ID] >= 0
            ):
                local_trees.append(
                    self.compartment_list[parent_id][NODE_TREE_ID]
                )
            for child_id in segment[NODE_CHILDREN]:
                child = self.compartment_list[child_id]
                if child[NODE_TREE_ID] >= 0:
                    local_trees.append(child[NODE_TREE_ID])

            if len(local_trees) == 0:
                tree_num = len(trees)
            elif len(local_trees) == 1:
                tree_num = local_trees[0]
            else:
                tree_num = local_trees[0]
                for local_tree in local_trees[1:]:
                    trees[local_tree] = []
                    for node in self.compartment_list:
                        if node[NODE_TREE_ID] == local_tree:
                            node[NODE_TREE_ID] = tree_num

            while len(trees) <= tree_num:
                trees.append([])
            trees[tree_num].append(segment)
            segment[NODE_TREE_ID] = tree_num

        self._tree_list = [tree for tree in trees if tree]

        soma_tree = -1
        for segment in self.compartment_list:
            if segment[NODE_TYPE] == Morphology.SOMA:
                soma_tree = segment[NODE_TREE_ID]
                break
        if soma_tree > 0:
            self._tree_list[soma_tree], self._tree_list[0] = (
                self._tree_list[0],
                self._tree_list[soma_tree],
            )
        self._reset_tree_ids()

    def _reset_tree_ids(self):
        """Synchronize each compartment's tree identifier."""
        for tree_id, tree in enumerate(self._tree_list):
            for segment in tree:
                segment[NODE_TREE_ID] = tree_id

    def _check_consistency(self):
        """Return the number of structural consistency errors."""
        errors = 0
        num_nodes = self.num_nodes
        for segment in self.compartment_list:
            parent_id = segment[NODE_PN]
            if parent_id >= 0 and parent_id >= num_nodes:
                print(
                    "Parent for node %d is invalid (%d)"
                    % (segment[NODE_ID], parent_id)
                )
                errors += 1

        for tree_id in range(self.num_trees):
            tree = self.tree(tree_id)
            root = -1
            for node_id in range(len(tree)):
                if tree[node_id][NODE_PN] == -1:
                    if root >= 0:
                        print("Too many roots in tree %d" % tree_id)
                        errors += 1
                    root = node_id
            if root == -1:
                print("No root present in tree %d" % tree_id)
                errors += 1

        adoptees = self._find_type_boundary()
        for child in adoptees:
            if child[NODE_TYPE] == Morphology.AXON:
                parent_id = child[NODE_PN]
                while parent_id >= 0:
                    parent = self.compartment_list[parent_id]
                    if parent[NODE_TYPE] == Morphology.AXON:
                        print("Branch has multiple axon roots")
                        print(child)
                        print(parent)
                        errors += 1
                        break
                    parent_id = parent[NODE_PN]
        if errors > 0:
            print(
                "Failed consistency check: %d errors encountered" % errors
            )
        return errors

    def _find_type_boundary(self):
        """Return compartments whose parents have a different SWC type."""
        adoptees = []
        for node in self.compartment_list:
            parent = self.parent_of(node)
            if parent is None:
                continue
            if node[NODE_TYPE] != parent[NODE_TYPE]:
                adoptees.append(node)
        return adoptees

    def save(self, file_name):
        """Write the morphology to an SWC file."""
        with open(file_name, "w") as swc_file:
            swc_file.write("#n,type,x,y,z,radius,parent\n")
            for segment in self.compartment_list:
                swc_file.write(
                    "%d %d " % (segment[NODE_ID], segment[NODE_TYPE])
                )
                swc_file.write("%0.4f " % segment[NODE_X])
                swc_file.write("%0.4f " % segment[NODE_Y])
                swc_file.write("%0.4f " % segment[NODE_Z])
                swc_file.write("%0.4f " % segment[NODE_R])
                swc_file.write("%d\n" % segment[NODE_PN])

    def write(self, file_name):
        """Write the morphology to an SWC file."""
        self.save(file_name)

    def _resolve_node_type(self, segment):
        """Resolve a compartment object or numeric index."""
        if not isinstance(segment, Compartment):
            try:
                segment = int(segment)
                if segment < 0 or segment >= len(self._compartment_list):
                    return None
                segment = self._compartment_list[segment]
            except ValueError:
                raise TypeError(
                    "Object not recognized as morphology node or index"
                )
        return segment

    def strip_type(self, node_type):
        """Remove compartments of one SWC type and rebuild the morphology."""
        flagged_for_removal = {}
        for segment in self.compartment_list:
            if segment[NODE_TYPE] == node_type:
                flagged_for_removal[segment[NODE_ID]] = True

        for index in range(len(self.compartment_list)):
            segment = self.compartment_list[index]
            if segment[NODE_ID] in flagged_for_removal:
                self.compartment_list[index] = None
            elif segment[NODE_PN] in flagged_for_removal:
                segment[NODE_PN] = -1
        self._reconstruct()

