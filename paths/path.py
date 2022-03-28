import copy


class Path:
    def __init__(self, nodes_involved, last_node):
        if isinstance(nodes_involved, set):
            self.nodes_involved = nodes_involved
        else:
            self.nodes_involved = set(nodes_involved)
        self.last_node = last_node

    def get_next_paths(self, adj_list, node_to_label_dictionary, my_path_label):
        assert isinstance(my_path_label, tuple)

        # get neighbours of last node
        last_node_neighbours = adj_list[self.last_node]

        # remove the neighbours that are already in the path
        last_node_neighbours = list(
            filter(lambda neighbour: neighbour not in self.nodes_involved, last_node_neighbours))

        new_paths = [Path(self.nodes_involved | {new_node}, new_node) for new_node in last_node_neighbours]

        # we can always assume that the selected node are present in the dictionary node to label because they are part of the graph
        last_node_neighbours_labels = [node_to_label_dictionary[node] for node in last_node_neighbours]

        new_labels = [my_path_label + tuple([node_label]) for node_label in last_node_neighbours_labels]
        return new_labels, new_paths
