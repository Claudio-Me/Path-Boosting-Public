from classes.extended_path_boosting.extended_pattern_boosting import ExtendedPatternBoosting
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
from settings_for_extended_pattern_boosting import SettingsExtendedPatternBoosting

from data import data_reader
from sklearn.model_selection import train_test_split

dataset_original_graphs = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs_original")

# create the model that contains the boosting matrix
if False:
    directory = "/Users/popcorn/PycharmProjects/pattern_boosting/results/Xgb_step_200_max_path_length_99999_5_k_selection_graphs_gbtree/wrapped_boosting"

    wrapper_pattern_boosting: WrapperPatternBoosting = data_reader.load_data(directory=directory,
                                                                             filename="wrapper_pattern_boosting")

    selected_paths = wrapper_pattern_boosting.get_selected_paths()

    train_list_graph_nx, test_list_graph_nx = train_test_split(dataset_original_graphs, test_size=0.2, random_state=0)

    extended_pattern_boosting = ExtendedPatternBoosting(train_data=train_list_graph_nx, test_data=test_list_graph_nx,
                                                        selected_paths=selected_paths,
                                                        settings=SettingsExtendedPatternBoosting())

    data_reader.save_data(data=extended_pattern_boosting, filename='extended_path_boosting',
                          directory='extended_path_boosting', create_unique_subfolder=False)

    # save all the attributes
    if True:
        dict_of_interaction_constraints = extended_pattern_boosting.dict_of_interaction_constraints
        data_reader.save_data(data=dict_of_interaction_constraints,
                              filename='dict_of_interaction_constraints',
                              directory='extended_path_boosting', create_unique_subfolder=False)

        test_ebm_dataframe = extended_pattern_boosting.test_ebm_dataframe
        data_reader.save_data(data=test_ebm_dataframe, filename='test_ebm_dataframe',
                              directory='extended_path_boosting', create_unique_subfolder=False)

        selected_paths = extended_pattern_boosting.selected_paths
        data_reader.save_data(data=selected_paths, filename='selected_paths',
                              directory='extended_path_boosting', create_unique_subfolder=False)

        train_ebm_dataframe = extended_pattern_boosting.train_ebm_dataframe
        data_reader.save_data(data=train_ebm_dataframe, filename='train_ebm_dataframe',
                              directory='extended_path_boosting', create_unique_subfolder=False)

        train_bm_df = extended_pattern_boosting.train_bm_df
        data_reader.save_data(data=train_bm_df, filename='train_bm_df',
                              directory='extended_path_boosting', create_unique_subfolder=False)

        test_bm_df = extended_pattern_boosting.test_bm_df
        data_reader.save_data(data=test_bm_df, filename='test_bm_df',
                              directory='extended_path_boosting', create_unique_subfolder=False)

# load data
if True:
    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='dict_of_interaction_constraints',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    dict_of_interaction_constraints = data_reader.load_data(directory=save_directory)

    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='test_ebm_dataframe',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    test_ebm_dataframe = data_reader.load_data(directory=save_directory)

    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='selected_paths',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    selected_paths = data_reader.load_data(directory=save_directory)

    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='train_ebm_dataframe',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    train_ebm_dataframe = data_reader.load_data(directory=save_directory)

    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='train_bm_df',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    train_bm_df = data_reader.load_data(directory=save_directory)

    save_directory = data_reader.get_save_location(file_extension=".pkl", file_name='test_bm_df',
                                                   folder_relative_path='extended_path_boosting',
                                                   unique_subfolder=False)
    test_bm_df = data_reader.load_data(directory=save_directory)

extended_pattern_boosting: ExtendedPatternBoosting = ExtendedPatternBoosting(
    dict_of_interaction_constraints=dict_of_interaction_constraints,
    selected_paths=selected_paths, train_boosting_matrix=train_bm_df,
    test_boosting_matrix=test_bm_df, train_data=train_ebm_dataframe, test_data=test_ebm_dataframe,
    settings=SettingsExtendedPatternBoosting())

extended_pattern_boosting.train()

settings_for_extended_pattern_boosting = SettingsExtendedPatternBoosting()
print(settings_for_extended_pattern_boosting)
