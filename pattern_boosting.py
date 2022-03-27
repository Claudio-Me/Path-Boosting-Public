from graph import GraphPB
from boosting_matrix import BoostingMatrix
from collections import defaultdict
import numpy as np

import networkx as nx
import itertools


class PatternBoosting:
    def __init__(self, dataset):
        # dataset is assumed to be a list of graphs
        if isinstance(dataset[0], GraphPB):
            self.dataset = dataset
        elif isinstance(dataset[0], nx.classes.multigraph.MultiGraph):
            self.dataset = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in dataset]
        else:
            raise TypeError("Graph format not recognized")

        self.pattern_boosting()

    def pattern_boosting(self):
        self.initialize_boosting_matrix()

    def initialize_boosting_matrix(self):

        # get a list of all the metal centers atomic numbers
        # metal_centers = list(itertools.chain(*[graph.metal_center for graph in self.dataset]))
        matrix_header = set()
        label_to_graphs = defaultdict(list)

        for i in range(len(self.dataset)):
            graph = self.dataset[i]
            metal_center_labels = graph.get_metal_center_labels()
            matrix_header.update(metal_center_labels)
            for label in metal_center_labels:
                label_to_graphs[label].append(int(i))

        boosting_matrix = np.zeros((len(self.dataset), len(matrix_header)), dtype=int)
        matrix_header = list(matrix_header)

        for ith_label in range(len(matrix_header)):
            label = matrix_header[ith_label]
            for ith_graph in label_to_graphs[label]:
                graph = self.dataset[ith_graph]
                nodes = graph.label_to_node[label]
                boosting_matrix[ith_graph][ith_label] = len(nodes)
                for node in nodes:
                    graph.selected_paths.add_path(path_label=label, path=[node])

        self.boosting_matrix = BoostingMatrix(boosting_matrix, matrix_header)
