import json


class Graph:

    def __init__(self, adjacency_matrix, node_labels):
        # adjacency matrix is assumed to be a boolean matrix
        self.adjacency_matrix = adjacency_matrix

        # nodel_labels is assumed to be a list where the i-th element is the label of the node i
        self.node_to_label = node_labels

        self.label_to_node = self.from_label_list_to_dictionary(node_labels)

    @staticmethod
    def from_label_list_to_dictionary(node_labels):
        nodes_tmp = list(range(len(node_labels)))
        my_dictionary = dict(zip(node_labels, nodes_tmp))

        return my_dictionary
