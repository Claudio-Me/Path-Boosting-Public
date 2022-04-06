from data import data_reader


class TestDatasetReading:
    def __init__(self):
        dataset = self.test_dataset_reading()
        pass
    def test_dataset_reading(self):
        return data_reader.read_data_from_directory("data/5k-selection-graphs")
