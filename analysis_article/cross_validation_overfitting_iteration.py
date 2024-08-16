import functools
import warnings

from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from classes.dataset import Dataset
from settings import Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from classes.pattern_boosting import PatternBoosting
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
import pandas as pd
import copy
from classes.analysis_patternboosting import AnalysisPatternBoosting
from data.load_dataset import load_dataset
from data import data_reader
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
import copy
from data import data_reader
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
from jupiter_notebook_functions import *
import warnings
from typing import List, Tuple, Optional
from classes.analysis_wrapper_pattern_boosting import AnalysisWrapperPatternBoosting
import random
import copy
from analysis_article.set_default_settings import set_default_settings


def cross_validation_synthetic_dataset(number_of_simulations=200, k_folds=5, synthetic_dataset_scenario=1,
                                       noise_variance=0.2,
                                       maximum_number_of_steps=None, save_fig=False):
    folder_relative_path = "results/jupiter/cross_validation"

    set_default_settings()

    Settings.noise_variance = noise_variance

    Settings.scenario = synthetic_dataset_scenario
    Settings.set_scenario(synthetic_dataset_scenario)

    Settings.save_analysis = False
    Settings.show_analysis = False
    Settings.dataset_name = "5k_synthetic_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    Settings.generate_new_dataset = True

    if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
        Settings.wrapper_boosting = False
    elif synthetic_dataset_scenario == 3:
        Settings.wrapper_boosting = True

    if maximum_number_of_steps is None:
        if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
            Settings.maximum_number_of_steps = 100
        elif synthetic_dataset_scenario == 3:
            Settings.maximum_number_of_steps = 300

    cros_validation_synthetic_dataset(folder_relative_path, n_iterations=number_of_simulations, k_folds=k_folds)

    # %%
    # Load data and print result
    Settings.scenario = 3
    Settings.set_scenario(Settings.scenario)

    folder_relative_path = "results/jupiter/cross_validation"
    directory = data_reader.get_save_location(folder_relative_path=folder_relative_path, unique_subfolder=False)

    list_overfitting_iterations = data_reader.load_data(directory=directory,
                                                        filename="list_overfitting_iterations_" + str(
                                                            Settings.noise_variance) + '_scenario_' + str(
                                                            Settings.scenario))
    list_of_test_errors = data_reader.load_data(directory=directory, filename="list_of_test_errors_" + str(
        Settings.noise_variance) + '_scenario_' + str(Settings.scenario))
    list_n_selected_paths = data_reader.load_data(directory=directory, filename="list_n_selected_paths_" + str(
        Settings.noise_variance) + '_scenario_' + str(Settings.scenario))

    print("average overfitting iteration:")
    print(np.average(list_overfitting_iterations[:200]))

    print("max overfitting iteration")
    print(max(list_overfitting_iterations))

    print("std overfitting iteration:")
    print(np.std(list_overfitting_iterations[:200]))

    print("average test error, std")
    final_test_error_list = []
    for test_error in list_of_test_errors:
        final_test_error_list.append(test_error[-1])

    print(np.average(final_test_error_list[:200]), np.std(final_test_error_list[:200]))

    print('n_selected paths, std ')
    number_of_selected_paths_list = []
    for path_list in list_n_selected_paths:
        number_of_selected_paths_list.append(len(path_list))

    print(np.average(number_of_selected_paths_list[:200]), np.std(number_of_selected_paths_list[:200]))

    plot_test_error_vs_iterations(list_of_test_errors, save_fig=True)


def cross_valudation_normal_dataset(number_of_simulations=200, synthetic_dataset_scenario=1, noise_variance=0.2,
                                    maximum_number_of_steps=None, save_fig=False):
    # settings

    Settings.maximum_number_of_steps = 400

    Settings.save_analysis = True
    Settings.show_analysis = False

    Settings.dataset_name = "5_k_selection_graphs"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    Settings.generate_new_dataset = False

    # in the error graph Print only the last N learners
    Settings.tail = 10000

    Settings.wrapper_boosting = False
    Settings.test_size = 0.2

    Settings.noise_variance = 0.2

    Settings.wrapper_boosting = True
    Settings.verbose = False

    dataset = load_dataset()

    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                      random_split_seed=Settings.random_split_test_dataset_seed)

    overfitting_iteration, test_error, n_selected_paths = perform_cross_validation(train_dataset, test_dataset, k=5,
                                                                                   random_seed=None, patience=3)

    name_addition = '_5_k_selection_graphs'
    folder_relative_path = "results/jupiter/cross_validation"

    directory = data_reader.get_save_location(folder_relative_path=folder_relative_path, unique_subfolder=False)

    save_data(data=[overfitting_iteration, test_error, n_selected_paths],
              names=['list_overfitting_iterations_' + name_addition, 'list_of_test_errors_' + name_addition,
                     'list_n_selected_paths_' + name_addition], directory=directory)

    # %%
    name_addition = '_5_k_selection_graphs'
    folder_relative_path = "results/jupiter/cross_validation"

    directory = data_reader.get_save_location(folder_relative_path=folder_relative_path, unique_subfolder=False)

    list_overfitting_iterations = data_reader.load_data(directory=directory,
                                                        filename='list_overfitting_iterations_' + name_addition)
    list_of_test_errors = data_reader.load_data(directory=directory, filename='list_of_test_errors_' + name_addition)
    list_n_selected_paths = data_reader.load_data(directory=directory,
                                                  filename='list_n_selected_paths_' + name_addition)

    print("overfitting_iterations")
    print(list_overfitting_iterations)

    print("test_errors iteration")
    print(list_of_test_errors)

    print("n_selected_paths")
    print(len(list_n_selected_paths))
