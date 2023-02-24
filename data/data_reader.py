from classes.dataset import Dataset
from settings import Settings

import pathlib
import os
import sys
import networkx as nx
import glob
import csv
import pandas as pd
import warnings
import pickle
from data.synthetic_dataset import SyntheticDataset


def read_data_from_name(dataset_name, directory="data/"):
    return nx.read_gml(directory + dataset_name)


def read_data_from_directory(directory):
    # directory can be relative directory
    print("reading dataset")
    if directory[-1] != "/":
        directory = directory + "/"
    names = glob.glob(directory + '*.gml')
    dataset = [None] * len(names)
    print(len(names))
    for i, name in enumerate(names):
        dataset[i] = read_data_from_name(name, directory="")

    # old reading version, very slow
    # dataset = [read_data_from_name(graph_name, directory="") for graph_name in names]
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


def save_dataset_in_binary_file(dataset, directory=None, filename: str = None):
    if directory is None:
        directory = "data/"
    if directory[-1] != "/":
        directory = directory + "/"
    if filename is None:
        filename = "binary_dataset"
    with open(directory + filename + ".pkl", 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)


def load_dataset_from_binary(directory=None, filename=None):
    if directory is None:
        directory = "data/"
    if directory[-1] != "/":
        directory = directory + "/"
    if filename is None:
        filename = "binary_dataset"

    with open(directory + filename + '.pkl', 'rb') as inp:
        dataset = pickle.load(inp)
    return dataset


def load_data(filename, directory=None):
    if directory is None:
        directory = "data/"
    if directory[-1] != "/":
        directory = directory + "/"

    with open(directory + filename + '.pkl', 'rb') as inp:
        data = pickle.load(inp)
    return data


def save_data(data, filename, directory="results"):
    if not (os.getcwd() in directory):
        directory = get_save_location(file_name=filename, file_extension=".pkl", folder_relative_path=directory)

    if not '.' in directory:
        directory = directory + filename + ".pkl"
    with open(directory, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)


def get_save_location(file_name: str = '', file_extension: str = '', folder_relative_path="results") -> str:
    location = os.path.join(os.getcwd(), folder_relative_path)
    if len(file_extension) > 0 and file_extension[0] != '.':
        raise TypeError("File extension must start with a dot")

    if location[-1] != '/':
        location = location + '/'

    if Settings.algorithm == "R":
        folder_name = "R_" + str(
            Settings.maximum_number_of_steps) + '_' + Settings.r_base_learner_name + '_' + Settings.family
    elif Settings.algorithm == "Full_xgb":
        folder_name = "Xgb_" + str(Settings.maximum_number_of_steps)
    elif Settings.algorithm == "Xgb_step":
        folder_name = "Xgb_step_" + str(Settings.maximum_number_of_steps)
    else:
        raise TypeError("Selected algorithm not recognized")

    folder_name = folder_name + "_max_path_length_" + str(Settings.max_path_length) + "_" + Settings.dataset_name
    if not os.path.exists(location + folder_name):
        os.makedirs(location + folder_name)
    folder_name = folder_name + "/"

    file_name = file_name.replace(" ", "_") + file_extension
    return location + folder_name + file_name


def load_dataset():
    if Settings.dataset_name == "5_k_selection_graphs":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="5_k_selection_graphs")
        else:
            print("Creating 5k dataset")
            dataset = read_data_from_directory("data/5k-selection-graphs")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset, filename="5_k_selection_graphs")
            return dataset

        return dataset
    elif Settings.dataset_name == "60k_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="60k_dataset")
        else:
            print("Creating 60k dataset")
            dataset = read_data_from_directory("data/dNatQ_graphs")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset, filename="60k_dataset")
        return dataset

    elif Settings.dataset_name == "5k_synthetic_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="5k_synthetic_dataset")

        else:
            print("Creating a new labels for 5k dataset")
            create_dataset = SyntheticDataset()
            dataset = create_dataset.create_dataset_from_5k_selection_graph()
            save_dataset_in_binary_file(dataset, filename="5k_synthetic_dataset")
        return dataset
