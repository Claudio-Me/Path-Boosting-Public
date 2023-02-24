from collections import defaultdict
import numpy as np
import networkx as nx
import warnings
import copy

from settings import Settings
from classes.paths.selected_paths import SelectedPaths


class GraphPB:
    # list of labels of metal center
    metal_labels = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                    72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                    104, 105, 106, 107, 108, 109, 110, 111, 112]  # fourth block

    def __init__(self, adjacency_matrix: np.ndarray, node_to_labels_dictionary: dict, label_value, adj_list=None):
        # adjacency matrix is assumed to be a boolean matrix
        self.adj_matrix = adjacency_matrix

        # nodel_labels is assumed to be a dictionary where the key is the node and the value is the feature
        self.node_to_label = node_to_labels_dictionary

        self.label_value = label_value

        self.label_to_node = self.__from_node_feature_dictionary_to_feature_node_dictionary(node_to_labels_dictionary)

        self.metal_center = self.find_metal_center_nodes()

        self.selected_paths = SelectedPaths()

        # adj_list is a dictionary
        if adj_list is None:
            self.adj_list = self.from_adj_matrix_to_adj_list()
        else:
            self.adj_list = adj_list

        # dictionary that is [(node,label)]-->[list of neighbours of "node" with label "label"]
        self.neighbours_with_label = self.create_dictionary_of_neighbours_with_label()

    def get_neighbours_of__with_label(self, node, label):
        # it returns the neighbours of node "node" that have label "label"
        return self.neighbours_with_label[(node, label)]

    def create_dictionary_of_neighbours_with_label(self):
        nl_dict = defaultdict(list)
        for node in self.adj_list:
            for neighbour in self.adj_list[node]:
                neighbour_label = self.node_to_label[neighbour]
                nl_dict[(node, neighbour_label)].append(neighbour)

        return nl_dict

    def get_new_paths_labels_and_add_them_to_the_dictionary(self, path_label: tuple):
        """
        it returns the possible extension of the input path that can be made in the graph
        note: if the input label is not present in the selected paths, an empty set is returned
        """
        return self.selected_paths.get_new_paths_labels_and_add_them_to_the_dictionary(path_label, self.adj_list,
                                                                                       self.node_to_label)

    def number_of_times_selected_path_is_present(self, path_label):
        """N.B. this function search only in "selected_paths" if the goal is to search in the whole graph then
        check the function number_of_time_path_is_present_in_graph implemented below"""
        return self.selected_paths.get_number_of_times_path_is_present(path_label)

    def get_label_of_node(self, node):
        return self.node_to_label[node]

    def get_nodes_with_label(self, label):
        return self.label_to_node[label]

    def get_metal_center_label_and_add_metal_center_to_selected_paths(self):
        warnings.warn("Metal list not initialized yet")
        metal_center_labels = []
        warning = True
        for metal_label in self.metal_labels:
            if metal_label in self.label_to_node:
                if warning and (len(self.label_to_node[metal_label]) > 1 or len(metal_center_labels) > 0):
                    warnings.warn("Warning found multiple candidates for metal center in the same molecule")
                    warning = False
                metal_center_labels.append(metal_label)
                self.selected_paths.add_path(metal_label, self.label_to_node[metal_label])

        if len(metal_center_labels) == 0:
            raise ValueError("Metal center not found")
        return metal_center_labels

    def get_metal_center_labels(self):
        metal_center_labels = []
        warning = True
        for metal_label in self.metal_labels:
            if metal_label in self.label_to_node:
                if warning and (len(self.label_to_node[metal_label]) > 1 or len(metal_center_labels) > 0):
                    warnings.warn("Warning found multiple candidates for metal center in the same molecule")
                    warning = False
                metal_center_labels = metal_center_labels + [[metal_label]]

        if len(metal_center_labels) == 0:
            raise ValueError("Metal center not found")
        return metal_center_labels

    def find_metal_center_nodes(self):
        warnings.warn("Metal list not initialized yet")
        metal_center = []
        warning = True
        for metal_label in self.metal_labels:
            if metal_label in self.label_to_node:
                if warning and (len(self.label_to_node[metal_label]) > 1 or len(metal_center) > 0):
                    warnings.warn("Warning found multiple candidates for metal center in the same molecule")
                    warning = False
                metal_center = metal_center + self.label_to_node[metal_label]

        if len(metal_center) == 0:
            raise ValueError("Metal center not found")
        return metal_center

    def from_adj_matrix_to_adj_list(self):
        adj_list = defaultdict(list)
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j] != 0:
                    adj_list[i].append(j)
        return adj_list

    def number_of_time_path_is_present_in_graph(self, path_label: tuple) -> int:
        """
        Takes in input a path label and returns the number of times this path is present in the graph not to be
        confused with the function "number_of_times_selected_path_is_present" that count only the already selected paths
        """
        starting_point = path_label[0]
        if not (starting_point in self.label_to_node):
            return 0
        else:
            result = [self.__find_path([], path_label, start_node) for start_node in self.label_to_node[starting_point]]
            return sum(result)

    def __find_path(self, old_visited_nodes: list, path_label: tuple, current_node) -> int:
        visited_nodes = copy.deepcopy(old_visited_nodes)
        visited_nodes.append(current_node)
        if len(path_label) == len(visited_nodes):
            # we covered all the path_label
            return 1
        elif (current_node, path_label[len(visited_nodes)]) in self.neighbours_with_label:
            new_nodes_list: list = []
            for new_node in self.neighbours_with_label[(current_node, path_label[len(visited_nodes)])]:

                if not (new_node in visited_nodes):
                    new_nodes_list.append(new_node)

            result = [self.__find_path(visited_nodes, path_label, new_node) for new_node in new_nodes_list]
            return sum(result)
        else:
            return 0

    @staticmethod
    def from_GraphNX_to_GraphPB(nx_Graph, label=None):
        # need to convert dictionary keys and values from string to integer
        n_t_l_d = nx.get_node_attributes(nx_Graph, 'feature_atomic_number')
        n_t_l_d = {int(k): v for k, v in n_t_l_d.items()}

        adj_matrix = np.array(nx.to_numpy_matrix(nx_Graph))

        adj_list = nx.to_dict_of_lists(nx_Graph)
        adj_list = {int(k): [int(i) for i in v] for k, v in adj_list.items()}
        if label is None:
            try:
                label = nx_Graph.graph[Settings.graph_label_variable]
            except:
                label = None

        return GraphPB(adjacency_matrix=adj_matrix, node_to_labels_dictionary=n_t_l_d, label_value=label,
                       adj_list=adj_list)

    @staticmethod
    def __from_node_feature_dictionary_to_feature_node_dictionary(node_to_feature_dictionary):
        my_pair_list = list(node_to_feature_dictionary.items())
        new_dict = defaultdict(list)
        for (value, key) in my_pair_list:
            new_dict[key].append(value)

        return new_dict

    @staticmethod
    def from_label_list_to_dictionary(node_labels_list):
        # given in input the list "node_labels" that contains in the i-th position the label for node `i`
        # returns a dictionary that associates to each label the list of nodes with that label

        my_pairs_list = list(zip(node_labels_list, range(len(node_labels_list))))

        new_dict = defaultdict(list)
        for (key, value) in my_pairs_list:
            new_dict[key].append(value)

        return new_dict

    def set_label(self, label):
        self.label = label
