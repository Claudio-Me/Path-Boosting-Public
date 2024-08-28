
from classes.extended_pattern_boosting import ExtendedPatternBoosting
from settings_for_extended_pattern_boosting import SettingsExtendedPatternBoosting

from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from data import data_reader
from classes.dataset import Dataset
import matplotlib as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


dataset_original_graphs = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs_original")

# load the trained model that contains the boosting matrix

directory = "/Users/popcorn/PycharmProjects/pattern_boosting/results/Xgb_step_300_max_path_length_101_5_k_selection_graphs/wrapped_boosting"

wrapper_pattern_boosting = data_reader.load_data(directory=directory, filename="wrapper_pattern_boosting")

selected_paths = wrapper_pattern_boosting.get_selected_paths()

extended_pattern_boosting = ExtendedPatternBoosting()

train_list_graph_nx, test_list_graph_nx = train_test_split(dataset_original_graphs, test_size=0.2, random_state=0)

extended_pattern_boosting.train(list_graphs_nx=train_list_graph_nx, test_data= test_list_graph_nx, selected_paths=selected_paths)


settings_for_extended_pattern_boosting=SettingsExtendedPatternBoosting()
print(settings_for_extended_pattern_boosting)