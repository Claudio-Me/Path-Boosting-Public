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


def signal_to_noise(number_of_simulations=200,
                    noise_variance_list=[0.325, 0.625, 0.875, 1.125, 1.375, 1.625, 0.2, 0.5, 0.75, 1, 1.25, 1.5],
                    synthetic_dataset_scenario=1,
                    dataset_name="5k_synthetic_dataset", noise_variance=0.2, maximum_number_of_steps=None,
                    save_fig=False, use_wrapper_boosting=None, show_settings=True):
    set_default_settings()

    Settings.noise_variance = noise_variance

    Settings.scenario = synthetic_dataset_scenario
    Settings.set_scenario(synthetic_dataset_scenario)

    Settings.save_analysis = False
    Settings.show_analysis = False
    Settings.dataset_name = dataset_name  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    if Settings.dataset_name == "5k_synthetic_dataset":
        Settings.generate_new_dataset = True
        fig_name = "signal_to_noise_ratio" + str(Settings.scenario) + ".pdf"
    else:
        Settings.generate_new_dataset = False
        fig_name = "signal_to_noise_ratio.pdf"

    Settings.wrapper_boosting = use_wrapper_boosting

    if (
            synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2) and Settings.dataset_name == "5k_synthetic_dataset":
        Settings.wrapper_boosting = False
    elif synthetic_dataset_scenario == 3 and Settings.dataset_name == "5k_synthetic_dataset":
        Settings.wrapper_boosting = True

    if maximum_number_of_steps is None:
        if synthetic_dataset_scenario == 1:
            Settings.maximum_number_of_steps = 28
        elif synthetic_dataset_scenario == 2:
            Settings.maximum_number_of_steps = 82
        elif synthetic_dataset_scenario == 3:
            Settings.maximum_number_of_steps = 290

    different_variances_final_test_error_vector = []
    different_variances_final_train_error_vector = []
    different_variances_missed_paths_counter = []
    different_variances_average_y_value = []

    for noise_variance in noise_variance_list:
        print("Noise Variance")
        print(noise_variance)
        Settings.noise_variance = noise_variance

        final_test_error_vector = []
        final_train_error_vector = []
        missed_paths_counter = []
        average_y_value = []
        for i in range(number_of_simulations):
            print("i")
            print(i)
            dataset = load_dataset()

            train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                              random_split_seed=Settings.random_split_test_dataset_seed)
            average_y_value.append(np.average(test_dataset.get_labels()))

            # pattern boosting
            pattern_boosting = PatternBoosting()
            pattern_boosting.training(train_dataset, test_dataset)
            final_test_error = pattern_boosting.test_error[-1]
            final_train_error = pattern_boosting.train_error[-1]
            final_test_error_vector.append(final_test_error)
            final_train_error_vector.append(final_train_error)

            selected_paths = pattern_boosting.get_selected_paths_in_boosting_matrix()
            synthetic_dataset = SyntheticDataset()
            missed_paths = []
            for target_path in synthetic_dataset.target_paths:
                if target_path not in selected_paths:
                    missed_paths.append(target_path)
            missed_paths_counter.append(len(missed_paths))

        different_variances_final_test_error_vector.append(final_test_error_vector)
        different_variances_final_train_error_vector.append(final_train_error_vector)
        different_variances_missed_paths_counter.append(missed_paths_counter)
        different_variances_average_y_value.append(np.average(average_y_value))

    if show_settings is True:
        Settings.print_principal_values()

    for i, variance in enumerate(noise_variance_list):
        print(variance)

        print("average test error")
        print(np.average(different_variances_final_test_error_vector[i]))

        print("standard error test error")
        print(np.std(different_variances_final_test_error_vector[i]))

        print("average train error")
        print(np.average(different_variances_final_train_error_vector[i]))

        print("standard error train error")
        print(np.std(different_variances_final_train_error_vector[i]))

        print("average missed_paths_counter")
        print(np.average(different_variances_missed_paths_counter[i]),
              np.std(different_variances_missed_paths_counter[i]))

        print("------------------------------------------------")

    # Calculate statistics based on simulation results
    mean_errors = []
    variance_errors = []
    min_errors = []
    max_errors = []

    # Assuming `simulation_results` is a list of lists (for each x, a list of 100 simulation errors)
    for x_simulation_errors in different_variances_final_test_error_vector:
        mean_errors.append(np.mean(x_simulation_errors))
        variance_errors.append(np.var(x_simulation_errors))
        min_errors.append(np.min(x_simulation_errors))
        max_errors.append(np.max(x_simulation_errors))

    plot_signal_to_noise_ratio(average_y_value=different_variances_average_y_value,
                               noise_variance_list=noise_variance_list,
                               variance_errors=variance_errors, mean_errors=mean_errors, min_errors=min_errors,
                               max_errors=max_errors, save_fig=save_fig, name_fig=fig_name)


signal_to_noise(number_of_simulations=50,
                noise_variance_list=[0.2, 0.325, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625],
                # [0.2, 0.325, 0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.625]
                synthetic_dataset_scenario=3,
                dataset_name="5k_synthetic_dataset", noise_variance=0.2, maximum_number_of_steps=None,
                save_fig=True, use_wrapper_boosting=None, show_settings=True)
