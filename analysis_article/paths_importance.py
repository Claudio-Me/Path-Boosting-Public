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


def paths_importance_analysis(dataset_name, number_of_simulations=200, synthetic_dataset_scenario=1, noise_variance=0.2,
                              maximum_number_of_steps=None, update_features_importance_by_comparison=True):
    set_default_settings()

    Settings.noise_variance = noise_variance

    Settings.scenario = synthetic_dataset_scenario
    Settings.set_scenario(synthetic_dataset_scenario)

    Settings.update_features_importance_by_comparison = update_features_importance_by_comparison

    Settings.save_analysis = False
    Settings.show_analysis = False
    Settings.dataset_name = dataset_name  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    Settings.generate_new_dataset = True

    if dataset_name == '5k_synthetic_dataset':
        if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
            Settings.wrapper_boosting = False
        elif synthetic_dataset_scenario == 3:
            Settings.wrapper_boosting = True

        if maximum_number_of_steps is None:
            if synthetic_dataset_scenario == 1:
                Settings.maximum_number_of_steps = 28
            if synthetic_dataset_scenario == 2:
                Settings.maximum_number_of_steps = 83
            elif synthetic_dataset_scenario == 3:
                Settings.maximum_number_of_steps = 300


    else:
        Settings.wrapper_boosting = True

    final_test_error_vector = []
    final_train_error_vector = []
    missed_paths_counter = []
    n_selected_paths = []
    n_selected_paths_per_iterations = []
    overfitting_iteration = []
    true_positive_ratio_1 = []
    selected_paths_set = set()
    cumulative_paths_importance = defaultdict(float)
    cumulative_times_selected = defaultdict(int)
    counts = defaultdict(int)
    dictionary_paths_importance_stored_in_lists = defaultdict(list)
    dictionary_n_times_selected_stored_in_lists = defaultdict(list)

    for i in range(number_of_simulations):
        print("i")
        print(i)
        dataset = load_dataset()

        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                          random_split_seed=Settings.random_split_test_dataset_seed)

        # pattern boosting
        pattern_boosting = PatternBoosting()
        pattern_boosting.training(train_dataset, test_dataset)
        final_test_error = pattern_boosting.test_error[-1]
        final_train_error = pattern_boosting.train_error[-1]
        final_test_error_vector.append(final_test_error)
        final_train_error_vector.append(final_train_error)
        n_selected_paths_per_iterations.append(pattern_boosting.n_selected_paths)
        true_positive_ratio_1.append(pattern_boosting.true_positive_ratio_1)
        selected_paths = pattern_boosting.get_selected_paths_in_boosting_matrix()
        n_selected_paths.append(len(selected_paths))
        # compute number of times a path is selected and average importance
        patterns_importance = pattern_boosting.get_boosting_matrix_normalized_columns_importance_values()

        for name, value in zip(pattern_boosting.get_boosting_matrix_header(), patterns_importance):
            if value > 0.0:
                cumulative_paths_importance[name] += value
                counts[name] += 1
                dictionary_paths_importance_stored_in_lists[name].append(value)

        for name, times_selected in zip(pattern_boosting.get_boosting_matrix_header(),
                                        pattern_boosting.get_number_of_times_path_has_been_selected()):
            if times_selected > 0:
                cumulative_times_selected[name] += times_selected
                dictionary_n_times_selected_stored_in_lists[name].append(times_selected)

        # compute overfitting iteration
        synthetic_dataset = SyntheticDataset()
        overfitting_iteration.append(early_stopping(test_errors=pattern_boosting.test_error, patience=3))
        missed_paths = []
        for target_path in synthetic_dataset.target_paths:
            if target_path not in selected_paths:
                missed_paths.append(target_path)
        missed_paths_counter.append(len(missed_paths))


    # add zeroes to every path list such that the length of the list for each path is equal to number of simulations
    for name in dictionary_paths_importance_stored_in_lists:
        if len(dictionary_paths_importance_stored_in_lists[name]) < number_of_simulations:
            zeroes = [0] * (number_of_simulations - len(dictionary_paths_importance_stored_in_lists[name]))
            dictionary_paths_importance_stored_in_lists[name] = dictionary_paths_importance_stored_in_lists[
                                                                    name] + zeroes


    for name in dictionary_n_times_selected_stored_in_lists:
        if len(dictionary_n_times_selected_stored_in_lists[name]) < number_of_simulations:
            zeroes = [0] * (number_of_simulations - len(dictionary_n_times_selected_stored_in_lists[name]))
            dictionary_n_times_selected_stored_in_lists[name] = dictionary_n_times_selected_stored_in_lists[
                                                                    name] + zeroes

    averages_importance = {name: (cumulative_paths_importance[name] / number_of_simulations,
                                  np.std(dictionary_paths_importance_stored_in_lists[name])) for name in
                           cumulative_paths_importance}
    averages_times_selected = {name: (
        cumulative_times_selected[name] / number_of_simulations,
        np.std(dictionary_n_times_selected_stored_in_lists[name]))
        for name in
        cumulative_times_selected}

    # Print averages values of results over synthetic dataset
    print("Averages importances")
    print_dict_sorted_by_values(averages_importance)

    print("average number of times selected")
    print_dict_sorted_by_values(averages_times_selected)

    print("final_test_error_vector")
    print(np.average(final_test_error_vector), np.std(final_test_error_vector))
    print("final_train_error_vector")
    print(np.average(final_train_error_vector), np.std(final_train_error_vector))
    print("n_selected_paths")
    print(np.average(n_selected_paths), np.std(n_selected_paths))

    # print("n_selected_paths_vector")
    # print(np.average(n_selected_paths_per_iterations, axis=0), np.std(n_selected_paths_per_iterations))

    print("overfitting_iteration")
    print(np.average(overfitting_iteration), np.std(overfitting_iteration))

    print("coefficients for the synthetic dataset")
    synthetic_dataset = SyntheticDataset()
    print(synthetic_dataset.target_paths)
    print(synthetic_dataset.coefficients)

    avg_selected_paths_per_iterations = np.average(n_selected_paths_per_iterations, axis=0)
    synthetic_dataset = SyntheticDataset()
    n_target_paths = len(synthetic_dataset.target_paths)

# uncomment to use the file as a script
paths_importance_analysis("5k_synthetic_dataset", number_of_simulations=200, synthetic_dataset_scenario=1, noise_variance=0.2, maximum_number_of_steps=None, update_features_importance_by_comparison=False)
