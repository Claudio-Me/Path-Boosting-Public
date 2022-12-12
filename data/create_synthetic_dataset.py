from data import data_reader


class CreateSyntheticDataset:
    def __init__(self):
        dataset = data_reader.load_dataset_from_binary(filename="60k_dataset")