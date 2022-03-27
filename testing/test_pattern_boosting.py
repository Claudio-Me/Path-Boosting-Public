import data.data_reader as dt
from graph import GraphPB
from pattern_boosting import PatternBoosting
import numpy as np


class TestPatternBoosting:
    def __init__(self, graph_dimension=3):
        GraphPB.metal_labels = [20]
        test_dataset = [self.create_test_graph(graph_dimension), self.create_test_graph(graph_dimension)]

        pattern_boosting = PatternBoosting(test_dataset)
        assert len(pattern_boosting.boosting_matrix) == 2
        assert len(pattern_boosting.boosting_matrix[0]) == 1

    def create_test_graph(self, graph_dimension):
        adjacency_matrix = np.ones((graph_dimension, graph_dimension)) - np.eye(graph_dimension)

        node_to_labels_dictionary = dict([(int(i), int(i) * 10) for i in range(graph_dimension)])
        return GraphPB(adjacency_matrix, node_to_labels_dictionary)
