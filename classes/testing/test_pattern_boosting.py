import data.data_reader as dt
from classes.graph import GraphPB
from classes.pattern_boosting import PatternBoosting
from classes.dataset import Dataset
from data import data_reader
from settings import Settings
from classes.splitted_training import SplittedTraining
import numpy as np


class TestPatternBoosting:
    def __init__(self):
        # self.test_1()
        # self.test_2()
        # self.test_on_n_dataset(100)
        # self.test_on_5k_dataset()
        # self.test_on_5k_dataset_with_test_data()
        self.test_on_60k_dataset_with_test_data()
        # self.test_splitted_training_on_5k_dataset_with_test_data()

    def test_1(self):
        print("testing patterboosting on fully connected graphs")
        # Note: this test assumes that the R function returns always the column 0
        graph_dimension = 3
        metal_labels = [20]
        settings = Settings()
        settings.maximum_number_of_steps = 2

        test_dataset = [self.__create_test_fully_connected_graph(graph_dimension, metal_labels),
                        self.__create_test_fully_connected_graph(graph_dimension, metal_labels)]

        pattern_boosting = PatternBoosting(settings)
        pattern_boosting.training(test_dataset)
        assert len(pattern_boosting.boosting_matrix.matrix) == 2
        assert len(pattern_boosting.boosting_matrix.matrix[0]) == 3
        assert np.count_nonzero(pattern_boosting.boosting_matrix.matrix) == 6
        labels_in_header = [tuple([20]), tuple([20, 0]), tuple([20, 10])]
        for label in labels_in_header:
            assert label in pattern_boosting.boosting_matrix.header

    def test_2(self):
        print("Testing patterbosting on 2-elelements dataset")
        LALMER_graph = dt.read_data_from_name("LALMER.gml")
        OREDIA_graph = dt.read_data_from_name("OREDIA.gml")
        dataset = [LALMER_graph, OREDIA_graph]
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(dataset)

    def test_on_n_dataset(self, n):
        print("Testing patternboosting on ", n, " test data")
        # --------------------------------------
        # dataset = data_reader.read_data_from_directory("data/5k-selection-graphs")
        # dataset = Dataset(dataset)
        # data_reader.save_dataset_in_binary_file(dataset)
        # -------------------------------------
        dataset = data_reader.load_dataset_from_binary()
        dataset = dataset.get_first_n_entries(n)
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(train_dataset, test_dataset)

    def test_on_5k_dataset(self):
        print("Testing patternboosting on 5k test data")
        # dataset = data_reader.read_data_from_directory("data/5k-selection-graphs")
        dataset = data_reader.load_dataset_from_binary()
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(dataset)

    def test_on_5k_dataset_with_test_data(self):
        print("Testing patternboosting on 5k data, with test data")
        # dataset = data_reader.read_data_from_directory("data/5k-selection-graphs")
        # dataset = Dataset(dataset)
        # data_reader.save_dataset_in_binary_file(dataset, filename="60k_dataset")
        dataset = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs")
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(train_dataset, test_dataset)

    def test_on_60k_dataset_with_test_data(self):
        print("Testing patternboosting on 60k data, with test data")
        # dataset = data_reader.read_data_from_directory("data/dNatQ_graphs")
        # dataset = Dataset(dataset)
        # data_reader.save_dataset_in_binary_file(dataset, filename="60k_dataset")
        dataset = data_reader.load_dataset_from_binary(filename="60k_dataset")
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(train_dataset, test_dataset)





    def __create_test_fully_connected_graph(self, graph_dimension, metal_labels):
        adjacency_matrix = np.ones((graph_dimension, graph_dimension)) - np.eye(graph_dimension)

        node_to_labels_dictionary = dict([(int(i), int(i) * 10) for i in range(graph_dimension)])
        graph = GraphPB(adjacency_matrix, node_to_labels_dictionary, label_value=0)

        graph.metal_labels = metal_labels
        return graph
