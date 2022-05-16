import data.data_reader as dt
from graph import GraphPB
import numpy as np


class TestGraphPBClass:
    def __init__(self, graph_dimension=3):
        self.create_test_graph(graph_dimension)

        self.test_selected_paths()

    def create_test_graph(self, graph_dimension):
        LALMER_graph = dt.read_data_from_name("LALMER.gml")

        self.test_graph_from_original_dataset = GraphPB.from_GraphNX_to_GraphPB(LALMER_graph)
        adjacency_matrix = np.ones((graph_dimension, graph_dimension)) - np.eye(graph_dimension)

        node_to_labels_dictionary = dict([(int(i), int(i) * 10) for i in range(graph_dimension)])
        self.my_test_graph = GraphPB(adjacency_matrix, node_to_labels_dictionary)

    def test_selected_paths(self):
        self.create_test_graph(3)
        pass