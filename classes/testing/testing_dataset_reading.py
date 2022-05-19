from data import data_reader


class TestDatasetReading:
    def __init__(self):
        graph = self.test_one_graph_read()
        dataset = self.test_dataset_reading()
        pass

    def test_dataset_reading(self):
        return data_reader.read_data_from_directory("data/5k-selection-graphs")

    def test_one_graph_read(self):
        return data_reader.read_data_from_name("LALMER.gml")
