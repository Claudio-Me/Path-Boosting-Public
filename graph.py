from collections import defaultdict
import numpy as np
import networkx as nx


class EssentialGraph:
    # list of labels of metal center
    metal_lables = []

    def __init__(self, adjacency_matrix, node_labels):
        # adjacency matrix is assumed to be a boolean matrix
        self.adjacency_matrix = adjacency_matrix

        # nodel_labels is assumed to be a list where the i-th element is the label of the node i
        self.node_to_label = node_labels

        self.label_to_node = self.from_label_list_to_dictionary(node_labels)

        self.metal_center = self.find_metal_center()

    def find_metal_center(self):
        pass

    @staticmethod
    def from_label_list_to_dictionary(node_labels):
        # given in input the array "node_labels" that contains in the i-th position the label for node `i`
        # returns a dictionary that associates to each label the list of nodes with that label

        my_pairs_list = list(zip(node_labels, range(len(node_labels))))

        new_dict = defaultdict(list)
        for (key, value) in my_pairs_list:
            new_dict[key].append(value)

        return new_dict
