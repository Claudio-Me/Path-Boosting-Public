from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from data import data_reader
from classes.dataset import Dataset
import matplotlib as plt
import pandas as pd
import numpy as np

# read the original dataset:
# dataset_original_graphs = data_reader.read_data_from_directory("data/5k-selection-graphs")

# data_reader.save_dataset_in_binary_file(dataset_original_graphs, filename="5_k_selection_graphs_original")

dataset_original_graphs = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs_original")

# load the trained model that contains the boosting matrix


directory = "/Users/popcorn/PycharmProjects/pattern_boosting/results/Xgb_step_300_max_path_length_101_5_k_selection_graphs/wrapped_boosting"

wrapper_pattern_boosting = data_reader.load_data(directory=directory, filename="wrapper_pattern_boosting")

selected_paths = wrapper_pattern_boosting.get_selected_paths()

extended_boosting_matrix = ExtendedBoostingMatrix()
extended_boosting_matrix.create_extend_boosting_matrix(selected_paths=selected_paths,
                                                       dataset=dataset_original_graphs)

# save the matrix
df: pd.DataFrame = extended_boosting_matrix.get_pandas_dataframe()

file_location = data_reader.get_save_location(file_name='extended_boosting_matrix', file_extension='.pkl',
                                              folder_relative_path='./results/extended_boosting_matrix',
                                              unique_subfolder=True)

df.to_pickle(file_location)

df = pd.read_pickle(file_location)

extended_boosting_matrix.plot_sparsity_matrix()
