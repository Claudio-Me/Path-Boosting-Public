from data import data_reader
from classes.dataset import Dataset
from data.synthetic_dataset import SyntheticDataset
from settings import Settings


class TestDatasetReading:
    def __init__(self):
        # graph = self.test_one_graph_read()
        # dataset = self.test_dataset_reading()
        # dataset = self.test_csv_read()
        # dataset = self.test_read_huge_dataset()
        new_synthetic_dataset = self.test_create_synthetic_dataset_from_5k_selection_graphs()
        pass

    def test_create_synthetic_dataset_from_5k_selection_graphs(self):
        print("Testing creating a new labels for 5k dataset")
        settings = Settings()
        create_dataset = SyntheticDataset(settings=settings)
        new_dataset = create_dataset.create_dataset_from_5k_selection_graph()
        data_reader.save_dataset_in_binary_file(new_dataset, filename="5k_synthetic_dataset")
        return new_dataset

    def test_dataset_reading(self):
        print("Test read 5k dataset")
        dataset = data_reader.read_data_from_directory("data/5k-selection-graphs")
        print("test done")
        return dataset

    def test_one_graph_read(self):
        print("Test read one graph")
        graph = data_reader.read_data_from_name("LALMER.gml")
        print("test done")
        return graph

    def test_csv_read(self):
        print("Test read dataset and csv")
        settings = Settings()
        dataset = data_reader.read_dataset_and_labels_from_csv("data/5k-selection-graphs", "tmQMg_5k_bin_class.csv", settings = settings)
        print("test done")
        return dataset

    def test_read_huge_dataset(self):
        print("Testing reading on 60k dataset")
        dataset = data_reader.read_data_from_directory("data/dNatQ_graphs")
        settings = Settings()
        dataset = Dataset(graphs_list=dataset, settings=settings)
        data_reader.save_dataset_in_binary_file(dataset, filename="60k_dataset")
        return data_reader.load_dataset_from_binary(filename="60k_dataset")
