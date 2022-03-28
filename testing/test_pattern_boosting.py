import data.data_reader as dt
from graph import GraphPB
from pattern_boosting import PatternBoosting
from settings import Settings
import numpy as np


class TestPatternBoosting:
    def __init__(self):
        self.test_1()

    def create_test_fully_connected_graph(self, graph_dimension, metal_labels):
        adjacency_matrix = np.ones((graph_dimension, graph_dimension)) - np.eye(graph_dimension)

        node_to_labels_dictionary = dict([(int(i), int(i) * 10) for i in range(graph_dimension)])
        graph = GraphPB(adjacency_matrix, node_to_labels_dictionary)

        graph.metal_labels = metal_labels
        return graph

    def test_1(self):
        # Note: this test assumes that the R function returns always the column 0
        graph_dimension = 3
        metal_labels = [20]
        settings = Settings()
        settings.number_of_learners = 2

        test_dataset = [self.create_test_fully_connected_graph(graph_dimension, metal_labels),
                        self.create_test_fully_connected_graph(graph_dimension, metal_labels)]

        pattern_boosting = PatternBoosting(test_dataset, settings)
        assert len(pattern_boosting.boosting_matrix.matrix) == 2
        assert len(pattern_boosting.boosting_matrix.matrix[0]) == 3
        assert np.count_nonzero(pattern_boosting.boosting_matrix.matrix) == 6
        labels_in_header = [tuple([20]), tuple([20, 0]), tuple([20, 10])]
        for label in labels_in_header:
            assert label in pattern_boosting.boosting_matrix.header
