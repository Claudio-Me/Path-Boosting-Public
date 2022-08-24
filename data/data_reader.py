from classes.dataset import Dataset

import networkx as nx
import glob
import csv
import pandas as pd
import warnings


def read_data_from_name(dataset_name, directory="data/"):
    return nx.read_gml(directory + dataset_name)


def read_data_from_directory(directory):
    print("reading dataset")
    if directory[-1] != "/":
        directory = directory + "/"
    names = glob.glob(directory + '*.gml')
    dataset = [read_data_from_name(graph_name, directory="") for graph_name in names]
    print("dataset loaded")
    return dataset


def split_training_and_test(dataset, test_size, labels: list = None):
    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset, labels)
    train_dataset, test_dataset = dataset.split_dataset(test_size)
    return train_dataset, test_dataset


def read_dataset_and_labels_from_csv(directory, file_name):
    warnings.warn("Weak implementation")
    if directory[-1] != "/":
        directory = directory + "/"
    names = glob.glob(directory + '*.gml')
    names.sort()
    dataset = [read_data_from_name(graph_name, directory="") for graph_name in names]
    labels = []
    print("Reading Dataset")
    df = pd.read_csv(directory + file_name, header=0, delimiter=';')
    df = df.sort_values(by='CSD_code')

    for i in range(len(names)):

        if not (df.loc[i].at['CSD_code'] in names[i]):
            raise TypeError("Debug names and order of graphs in the dataset is not the same")
        if df.loc[i].at["Stability"] == "Unstable":
            labels.append(0)
        elif df.loc[i].at["Stability"] == "Stable":
            labels.append(1)
        else:
            raise TypeError("Value not recognized")
    dataset = Dataset(dataset, labels)
    print("dataset loaded")

    return dataset
