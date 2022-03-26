from graph import GraphPB
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
            metal_labels = self.dataset[i].get_metal_center_label_and_add_metal_center_to_selected_paths()
            matrix_header.update(metal_labels)
            for label in metal_labels:
                label_to_graphs[label].append(int(i))

        self.boosting_matrix = np.zeros((len(self.dataset), len(matrix_header)), dtype=int)
        self.matrix_header = list(matrix_header)

        for i in range(len(self.matrix_header)):
            label = self.matrix_header[i]
            if label in label_to_graphs:
                self.boosting_matrix[i][label_to_graphs[label]] = 1
