from graph import GraphPB
from boosting_matrix import BoostingMatrix
from R_code.interface_with_R_code import LaunchRCode
from settings import Settings

from collections import defaultdict
import numpy as np
import networkx as nx

import itertools


class PatternBoosting:
    def __init__(self, dataset, settings=Settings()):
        self.settings = settings

        # dataset is assumed to be a list of graphs
        if isinstance(dataset[0], GraphPB):
            self.dataset = dataset
        elif isinstance(dataset[0], nx.classes.multigraph.MultiGraph):
            self.dataset = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in dataset]
        else:
            raise TypeError("Graph format not recognized")

        self.pattern_boosting()

    def pattern_boosting(self):
        # class to launch code in R
        self.r_interface = LaunchRCode()

        self.initialize_boosting_matrix()

        for i in range(self.settings.number_of_learners - 1):

            selected_column_number = self.r_interface.launch_function(self.boosting_matrix.matrix)

            if not (selected_column_number in self.boosting_matrix.already_selected_columns):
                # if the selected column has never been selected before
                self.boosting_matrix.already_selected_columns.add(selected_column_number)

                selected_column = self.boosting_matrix.matrix[:, selected_column_number]
                selected_path_label = self.boosting_matrix.header[selected_column_number]

                graphs_that_contain_selected_column_path = np.nonzero(selected_column)[0]

                new_paths_labels = self.__get_new_paths(selected_path_label, graphs_that_contain_selected_column_path)
                new_columns = self.__get_new_columns(new_paths_labels, graphs_that_contain_selected_column_path)

                self.boosting_matrix.add_column(new_columns, new_paths_labels)

    def __get_new_columns(self, new_paths, graphs_that_contain_selected_column_path):

        new_columns = np.zeros((len(self.dataset), len(new_paths)))

        for path_number in range(len(new_paths)):
            path = new_paths[path_number]
            for graph_number in graphs_that_contain_selected_column_path:
                graph = self.dataset[graph_number]
                n = graph.number_of_times_selected_path_is_present(path)
                new_columns[graph_number][path_number] = n

        return new_columns

    def __get_new_paths(self, selected_path_label, graphs_that_contain_selected_column_path):

        new_paths = [list(
            self.dataset[graph_number].get_new_paths_labels_and_add_them_to_the_dictionary(selected_path_label))
            for graph_number in graphs_that_contain_selected_column_path]
        new_paths = list(set([path for paths_list in new_paths for path in paths_list]))
        return new_paths

    def initialize_boosting_matrix(self):

        # get a list of all the metal centers atomic numbers
        # metal_centers = list(itertools.chain(*[graph.metal_center for graph in self.dataset]))
        matrix_header = set()
        label_to_graphs = defaultdict(list)

        for i in range(len(self.dataset)):
            graph = self.dataset[i]
            metal_center_labels = graph.get_metal_center_labels()
            metal_center_labels = [tuple(label) for label in metal_center_labels]
            matrix_header.update(metal_center_labels)
            for label in metal_center_labels:
                label_to_graphs[label].append(int(i))

        boosting_matrix = np.zeros((len(self.dataset), len(matrix_header)), dtype=int)
        matrix_header = list(matrix_header)

        for ith_label in range(len(matrix_header)):
            label = matrix_header[ith_label]
            for ith_graph in label_to_graphs[label]:
                graph = self.dataset[ith_graph]
                nodes = graph.label_to_node[label[0]]
                boosting_matrix[ith_graph][ith_label] = len(nodes)
                for node in nodes:
                    graph.selected_paths.add_path(path_label=label, path=[node])

        self.boosting_matrix = BoostingMatrix(boosting_matrix, matrix_header)
