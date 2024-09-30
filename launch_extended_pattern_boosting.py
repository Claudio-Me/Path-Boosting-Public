from classes.extended_pattern_boosting import ExtendedPatternBoosting
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
from settings_for_extended_pattern_boosting import SettingsExtendedPatternBoosting
from data import data_reader

from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from data import data_reader
import pickle
from sklearn.model_selection import train_test_split

dataset_original_graphs = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs_original")

# load the trained model that contains the boosting matrix

if True:
    directory = "/Users/popcorn/PycharmProjects/pattern_boosting/results/Xgb_step_200_max_path_length_99999_5_k_selection_graphs_gbtree/wrapped_boosting"

    wrapper_pattern_boosting: WrapperPatternBoosting = data_reader.load_data(directory=directory,
                                                                             filename="wrapper_pattern_boosting")

    selected_paths = wrapper_pattern_boosting.get_selected_paths()

    train_list_graph_nx, test_list_graph_nx = train_test_split(dataset_original_graphs, test_size=0.2, random_state=0)

    extended_pattern_boosting = ExtendedPatternBoosting(train_data=train_list_graph_nx, test_data=test_list_graph_nx,
                                                        selected_paths=selected_paths,
                                                        settings=SettingsExtendedPatternBoosting())

    save_directory = data_reader.save_data(data=extended_pattern_boosting, filename='extended_path_boosting',                                           directory='extended_path_boosting', create_unique_subfolder=False)

save_directory = data_reader.get_save_location(file_name='extended_path_boosting', file_extension = '.pkl', folder_relative_path="extended_path_boosting",                       unique_subfolder=False)

extended_pattern_boosting: ExtendedPatternBoosting = data_reader.load_data(directory=save_directory)

extended_pattern_boosting.train()

settings_for_extended_pattern_boosting = SettingsExtendedPatternBoosting()
print(settings_for_extended_pattern_boosting)
