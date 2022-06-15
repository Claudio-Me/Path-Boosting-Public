from data import data_reader


class TestDatasetReading:
    def __init__(self):
        graph = self.test_one_graph_read()
        dataset = self.test_dataset_reading()
        dataset = self.test_csv_read()
        pass

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
        dataset = data_reader.read_dataset_and_labels_from_csv("data/5k-selection-graphs", "tmQMg_5k_bin_class.csv")
        print("test done")
        return dataset
