from classes.dataset import Dataset

import networkx as nx
import glob


def read_data_from_name(dataset_name, directory="data/"):
    return nx.read_gml(directory + dataset_name)


def read_data_from_directory(directory):
    if directory[-1] != "/":
        directory = directory + "/"
    names = glob.glob(directory + '*.gml')
    dataset = [read_data_from_name(graph_name, directory="") for graph_name in names]
    print("dataset loaded")
    return dataset


def split_training_and_test(dataset, test_size):
    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset)
    train_dataset, test_dataset = dataset.split_dataset(test_size)
    return train_dataset, test_dataset
