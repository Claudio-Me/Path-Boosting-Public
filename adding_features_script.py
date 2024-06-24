from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from data.data_reader import read_data_from_directory, save_dataset_in_binary_file
from classes.dataset import Dataset

dataset = read_data_from_directory("data/5k-selection-graphs")

save_dataset_in_binary_file(dataset, filename="5_k_selection_graphs_original")


extended_boosting_matrix = ExtendedBoostingMatrix()
