from collections import defaultdict

from classes.dataset import Dataset

from settings import Settings
import math
import pathlib
import os
import sys
import networkx as nx
import glob
import csv
import pandas as pd
import warnings
import pickle
from typing import Tuple
from pathlib import Path, PosixPath
import copy


def read_data_from_name(dataset_name, directory="data/"):
    return nx.read_gml(directory + dataset_name)


def read_data_from_directory(directory):
    # directory can be relative directory
    print("reading dataset")
    if isinstance(directory, PosixPath):
        directory = str(directory)
    if directory[-1] != "/":
        directory = directory + "/"
    names = []
    for x in os.listdir(directory):
        if x.endswith(".gml"):
            names.append(x)
    # names = glob.glob(directory + '*.gml')
    dataset = [None] * len(names)
    print(len(names))
    for i, name in enumerate(names):
        dataset[i] = read_data_from_name(name, directory=directory)

    # old reading version, very slow
    # dataset = [read_data_from_name(graph_name, directory="") for graph_name in names]
    print("dataset loaded")
    return dataset


def split_training_and_test(dataset, test_size, labels: list = None, random_split_seed=None) -> Tuple[Dataset, Dataset]:
    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset, labels)
    if Settings.dataset_name == "5k_synthetic_dataset":
        # we have to verify that in the splitting all the target paths are in the training dataset at least once
        train_dataset, test_dataset = dataset.split_dataset(test_size, random_split_seed)

        found_target_paths = target_paths_contained_in_dataset(train_dataset)

        if len(found_target_paths) == len(Settings.target_paths):
            return train_dataset, test_dataset
        else:
            # not all target paths are in the train dataset
            target_paths_not_found_in_train_dataset = set(Settings.target_paths) - found_target_paths
            for path in target_paths_not_found_in_train_dataset:
                graphs_containing_path = []
                for graph in test_dataset.get_graphs_list():
                    if graph.number_of_time_path_is_present_in_graph(path) > 0:
                        graphs_containing_path.append(graph)

                number_of_graphs_to_be_moved_to_train_dataset = math.ceil(len(graphs_containing_path) / 2)
                for i in range(number_of_graphs_to_be_moved_to_train_dataset):
                    train_dataset.add(graphs_containing_path[i])
                    test_dataset.remove(graphs_containing_path[i])

            return train_dataset, test_dataset


    else:
        train_dataset, test_dataset = dataset.split_dataset(test_size, random_split_seed)
    return train_dataset, test_dataset


def target_paths_contained_in_dataset(dataset) -> set:
    # takes as input a dataset and it returns all the target paths that have been found in the graphs of the dataset
    found_target_paths = set()
    for path in Settings.target_paths:
        for graph in dataset.get_graphs_list():
            if graph.number_of_time_path_is_present_in_graph(path) > 0:
                found_target_paths.add(path)
                break
    return found_target_paths


def read_dataset_and_labels_from_csv(directory, file_name):
    directory = str(directory)
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
    directory = str(directory)
    if directory[-1] != "/":
        directory = directory + "/"
    if filename is None:
        filename = "binary_dataset"
    with open(directory + filename + ".pkl", 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)


def load_dataset_from_binary(directory=None, filename=None):
    if directory is None:
        directory = "data/"
    directory = str(directory)
    if directory[-1] != "/":
        directory = directory + "/"
    if filename is None:
        filename = "binary_dataset"

    with open(directory + filename + '.pkl', 'rb') as inp:
        dataset = pickle.load(inp)
    return dataset


def load_data(filename=None, directory=None):
    if directory is None:
        directory = "data/"
    directory = str(directory)
    if filename is not None:
        if directory[-1] != "/":
            directory = directory + "/"
        directory = directory + filename + '.pkl'
    with open(directory, 'rb') as inp:
        data = pickle.load(inp)
    return data


def save_data(data, filename, directory="results", create_unique_subfolder=True)-> str:
    directory = str(directory)
    directory_of_this_file = str(Path(__file__).parent.resolve().parent)
    if not (directory_of_this_file in directory):
        directory = get_save_location(file_name=filename, file_extension=".pkl", folder_relative_path=directory,
                                      unique_subfolder=create_unique_subfolder)

    if not '.' in directory:
        directory = directory + filename + ".pkl"
    with open(directory, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return directory

def get_save_location(file_name: str = '', file_extension: str = '', folder_relative_path="results",
                      unique_subfolder=False) -> str:
    # make sure that we are in the folder "pattern_boosting"
    last_folder = os.path.basename(os.path.normpath(os.getcwd()))
    if last_folder == "pattern_boosting":
        location = os.getcwd()
    elif last_folder == "classes" or last_folder == "data":
        # it removes the last directory from os.getcwd()
        location = os.path.dirname(os.getcwd())
    elif str(last_folder) == "analysis_article":
        # it removes the last directory from os.getcwd()
        location = os.path.dirname(os.getcwd())
    else:
        raise Exception("Uknown location ", last_folder)

    location = os.path.join(location, folder_relative_path)
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
    elif Settings.algorithm == "decision_tree":
        folder_name = "decision_tree_step_" + str(Settings.maximum_number_of_steps)
    else:
        raise TypeError("Selected algorithm not recognized")

    folder_name = folder_name + "_max_path_length_" + str(
        Settings.max_path_length) + "_" + Settings.dataset_name + "_" + Settings.xgb_model_parameters['booster']

    if Settings.wrapper_boosting is True:
        folder_name = folder_name + "/wrapped_boosting"

    if (not os.path.exists(location + folder_name)) and (unique_subfolder is True):
        os.makedirs(location + folder_name)
    elif (not os.path.exists(location)) and (unique_subfolder is False):
        os.makedirs(location)

    folder_name = folder_name + "/"

    file_name = file_name.replace(" ", "_") + file_extension
    if unique_subfolder is True:
        return location + folder_name + file_name

    else:
        return location + file_name
