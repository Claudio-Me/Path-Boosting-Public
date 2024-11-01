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
from data import data_reader
import matplotlib.cm as cm

import matplotlib.colors as mcolors

from xgboost import plot_importance

from sklearn import metrics
from classes.boosting_matrix import BoostingMatrix
from classes.dataset import Dataset
from settings import Settings
from matplotlib.ticker import MaxNLocator
from collections import Counter
from classes.pattern_boosting import PatternBoosting
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
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
import functools
import copy
from data import data_reader
import seaborn as sns

import pathlib
import os
import sys
from collections.abc import Iterable
from matplotlib.ticker import MaxNLocator
from typing import List
from classes.dataset import Dataset
import random
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import multiprocessing as mp


def get_XGB_error_and_variable_importance_t(max_path_length, pattern_boosting, max_number_of_learners, frequency_matrix,
                                            xgb_settings=None):
    header = list(frequency_matrix.columns)

    columns_to_be_removed = []
    print("Setting up matrix ready")
    for i, path in enumerate(header):
        if len(path) > max_path_length:
            columns_to_be_removed.append(path)
    print("Frequency matrix ready")

    if isinstance(pattern_boosting, PatternBoosting):
        x_test = pattern_boosting.create_boosting_matrix_for(pattern_boosting.test_dataset, selected_paths=set(
            pattern_boosting.boosting_matrix.get_selected_paths()))
    elif isinstance(pattern_boosting, WrapperPatternBoosting):
        x_test = pattern_boosting.create_ordered_boosting_matrix(pattern_boosting.test_dataset, selected_paths=set(
            pattern_boosting.get_selected_paths()))
    else:
        raise TypeError("Unknown pattern Boosting type")

    x_train = pattern_boosting.boosting_matrix.matrix
    y_test = pattern_boosting.test_dataset.labels
    y_train = pattern_boosting.training_dataset.labels

    if xgb_settings is None:
        xgb_settings = Settings.xgb_model_parameters

    xgb_settings['n_estimators'] = max_number_of_learners
    xgb_model = XGBRegressor(**xgb_settings, base_score=np.mean(y_train))

    eval_set = [(x_train, y_train), (x_test, y_test)]
    xgb_model.fit(x_train, y_train, eval_set=eval_set)

    # test the model
    # y_test_pred = xgb_model.predict(x_test)
    # y_train_pred = xgb_model.predict(x_train)

    if Settings.final_evaluation_error == "MSE":
        results = xgb_model.evals_result()
        train_error = results['validation_0'][Settings.xgb_model_parameters["eval_metric"]]
        test_error = results['validation_1'][Settings.xgb_model_parameters["eval_metric"]]
        # settings eval is rmse so we take the square
        train_error = [error ** 2 for error in train_error]
        test_error = [error ** 2 for error in test_error]

        # model_test_error = metrics.mean_squared_error(y_test, y_test_pred)
        # model_train_error = metrics.mean_squared_error(y_train, y_train_pred)

    else:
        raise ValueError("measure error not found")

    # plot_importance(xgb_model)
    # plt.show()
    print("max path length: ", max_path_length)
    return test_error, train_error


def get_XGB_error_and_variable_importance(max_path_length, frequency_matrix, labels, max_number_of_learners,
                                          xgb_settings=None):
    header = list(frequency_matrix.columns)

    columns_to_be_removed = []
    for i, path in enumerate(header):
        if len(path) > max_path_length:
            columns_to_be_removed.append(path)
    x_train, x_test, y_train, y_test = train_test_split(frequency_matrix.drop(columns=columns_to_be_removed),
                                                        labels,
                                                        test_size=Settings.test_size,
                                                        random_state=Settings.random_split_test_dataset_seed)

    if xgb_settings is None:
        xgb_settings = Settings.xgb_model_parameters

    # for i in range(1, max_number_of_learners + 1,10):

    xgb_settings['n_estimators'] = max_number_of_learners
    xgb_model = XGBRegressor(**xgb_settings, base_score=np.mean(y_train))

    eval_set = [(x_train, y_train), (x_test, y_test)]
    xgb_model.fit(x_train, y_train, eval_set=eval_set)

    # test the model
    # y_test_pred = xgb_model.predict(x_test)
    # y_train_pred = xgb_model.predict(x_train)

    if Settings.final_evaluation_error == "MSE":
        results = xgb_model.evals_result()
        train_error = results['validation_0'][Settings.xgb_model_parameters["eval_metric"]]
        test_error = results['validation_1'][Settings.xgb_model_parameters["eval_metric"]]
        # settings eval is rmse so we take the square
        train_error = [error ** 2 for error in train_error]
        test_error = [error ** 2 for error in test_error]

        # model_test_error = metrics.mean_squared_error(y_test, y_test_pred)
        # model_train_error = metrics.mean_squared_error(y_train, y_train_pred)

    else:
        raise ValueError("measure error not found")

    # plot_importance(xgb_model)
    # plt.show()
    print("max path length: ", max_path_length)
    return test_error, train_error


def is_sub_tuple(s, l):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False

    else:
        for i in range(len(l)):
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i + n] == s[n]):
                    n += 1

                if n == len(s):
                    sub_set = True

    return sub_set


def get_ratio_with_next_paths(parent_index, parent_value, presence_per_observation, ratio, acccepred_error):
    founded_paths = []
    ratios = []
    for index, value in presence_per_observation.items():
        if parent_index == index:
            continue
        elif is_sub_tuple(parent_index, index):
            ratio_here = value / parent_value
            if ratio_here - acccepred_error < ratio < ratio_here + acccepred_error:
                ratios.append(ratio_here)
                founded_paths.append(index)
    return ratios, founded_paths


def ceildiv(a, b):
    return -(a // -b)


# Assumes y2 is a list of list
def plot_graphs_new(x, y, tittle: str, x_label: str = "", y_label: str = "", show=True, save=True, y2=None, x2=None,
                    max_path_length=[]):
    plt.style.use('ggplot')
    if x2 == None:
        x2 = x
    fig, ax = plt.subplots()
    ax.set_ylim(0.00075, 0.000125)
    # Using set_dashes() to modify dashing of an existing line
    if len(x) > Settings.tail:
        ax.plot(x[-Settings.tail:], y[-Settings.tail:], label='')
    else:
        ax.plot(x, y, label='Pattern Boosting')
        for i, max_length in enumerate(max_path_length):
            ax.plot(x2, y2[i], label='XGB_' + str(max_length))

    ax.legend()
    ax.set_title(tittle)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    # plt.grid()

    # plot only integers on the x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    saving_location = data_reader.get_save_location(tittle, '.png')

    if save is True:
        plt.savefig(saving_location)
    if show is True:
        plt.show()

    return fig, ax


def plot_graphs(x, y, tittle: str, x_label: str = "", y_label: str = "", show=True, save=True, y2=None, x2=None,
                y_lim: tuple = None):
    plt.style.use('ggplot')

    fig, ax = plt.subplots()
    if y_lim is None:
        ax.set_ylim(0.00075, 0.00125)
    # Using set_dashes() to modify dashing of an existing line
    if len(x) > Settings.tail:
        ax.plot(x[-Settings.tail:], y[-Settings.tail:], label='')
    else:
        ax.plot(x, y, label='Pattern Boosting')
        ax.plot(x2, y2, label='XGB')

    ax.legend()
    ax.set_title(tittle)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    # plt.grid()

    # plot only integers on the x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    saving_location = data_reader.get_save_location(tittle, '.png')

    if save is True:
        plt.savefig(saving_location)
    if show is True:
        plt.show()

    return fig, ax


def plot_graphs_new_temp_funct(x, y, tittle: str, max_number_of_learners, test_err_full_power_xgb, xgb_settings,
                               test_err_full_power_xgb_on_pattern_boosting_matrix, x_label: str = "", y_label: str = "",
                               show=True, save=True, y2=None, x2=None, max_path_length=[]):
    plt.style.use('ggplot')
    if x2 == None:
        x2 = x
    fig, ax = plt.subplots()

    # Using set_dashes() to modify dashing of an existing line
    if len(x) > Settings.tail:
        ax.plot(x[-Settings.tail:], y[-Settings.tail:], label='')
    else:
        ax.plot(x, y, label='Pattern Boosting')
        for i, max_length in enumerate(max_path_length):
            ax.plot(x2, y2[i], label='XGB_' + str(max_length))

    ax.plot(max_number_of_learners, test_err_full_power_xgb, label='XGB depth ' + str(xgb_settings['max_depth']))
    ax.plot(max_number_of_learners, test_err_full_power_xgb_on_pattern_boosting_matrix, label="XGB on pb matrix")

    # plt.grid()
    ax.legend()
    ax.set_title(tittle)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    # plot only integers on the x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    saving_location = data_reader.get_save_location(tittle, '.png')

    if save is True:
        plt.savefig(saving_location)
    if show is True:
        plt.show()

    return fig, ax


def early_stopping(test_errors, patience=5):
    """
    Implements early stopping to detect overfitting.

    Args:
    test_errors (list of float): A list of test errors at each iteration.
    patience (int): Number of epochs to wait before stopping after validation error increases.

    Returns:
    int: The iteration number where overfitting starts.
         If no overfitting is detected, returns -1.
    """
    best_val_error = float('inf')
    best_iteration = -1
    count = 0

    for i in range(len(test_errors)):
        if test_errors[i] < best_val_error:
            best_val_error = test_errors[i]
            best_iteration = i
            count = 0  # Reset count if new best is found
        else:
            count += 1
            if count >= patience:
                return best_iteration

    return best_iteration


def plot_tpr_vs_iterations_max_min(true_positive_ratios_per_iteration: list[list[float]], save_fig=True):
    """
    Plots the true positive ratio with standard deviation error bars, max, and min
    against the number of iterations.
    The iterations are inferred based on the length of the true_positive_ratios list.

    :param true_positive_ratios_per_iteration: A list of list containing the TPR values at each iteration for each simulation.
    :type true_positive_ratios_per_iteration: list of list of float
    :returns: None
    """
    # Convert to numpy array for easier manipulation
    np_tpr_per_iteration = np.array(true_positive_ratios_per_iteration)

    # Calculate statistics
    mean_tpr = np.mean(np_tpr_per_iteration, axis=0)
    max_tpr = np.max(np_tpr_per_iteration, axis=0)
    min_tpr = np.min(np_tpr_per_iteration, axis=0)
    std_tpr = np.std(np_tpr_per_iteration, axis=0)  # Calculate the standard deviation

    iterations = list(range(1, len(mean_tpr) + 1))

    plt.figure(figsize=(10, 6))
    plt.errorbar(iterations, mean_tpr, yerr=std_tpr, fmt='-o', color='b',
                 label='True Positive Ratio (TPR) with Std. Dev.', ecolor='lightred', elinewidth=3, capsize=0)
    plt.fill_between(iterations, min_tpr, max_tpr, color='b', alpha=0.1, label='Min-Max Range')
    # plt.scatter(iterations, max_tpr, marker='^', color='g', label='Max TPR')
    # plt.scatter(iterations, min_tpr, marker='v', color='r', label='Min TPR')

    plt.xlabel('Iterations')
    plt.ylabel('True Positive Ratio')
    plt.title('True Positive Ratio vs. Iterations with Max, Min, and Std. Dev.')
    plt.legend()
    plt.grid(True)
    if save_fig is True:
        plt.savefig("true_positive_ratio_with_std_dev.pdf")
    plt.show()


def plot_tpr_vs_iterations(true_positive_ratios: list[float]):
    """
    Plots the true positive ratio against the number of iterations.
    The iterations are inferred based on the length of the true_positive_ratios list.

    :param true_positive_ratios: A list containing the TPR values at each iteration.
    :type true_positive_ratios: list of float
    :returns: None
    """
    iterations = list(range(1, len(true_positive_ratios) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, true_positive_ratios, marker='o', linestyle='-', color='b', label='True Positive Ratio (TPR)')
    plt.xlabel('Iterations')
    plt.ylabel('True Positive Ratio')
    plt.title('True Positive Ratio vs. Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig("true_positive_ratio.pdf")
    plt.show()


def plot_tpr_vs_iterations_different_definitions(tpr1: list[float], tpr2: list[float], tpr3: list[float]):
    """
    Plots the true positive ratio against the number of iterations for three different sets of TPR values.

    :param tpr1: A list containing the TPR values at each iteration for the first set.
    :type tpr1: list of float
    :param tpr2: A list containing the TPR values at each iteration for the second set.
    :type tpr2: list of float
    :param tpr3: A list containing the TPR values at each iteration for the third set.
    :type tpr3: list of float
    :returns: None
    """
    # Assuming all input lists have the same length
    iterations = list(range(1, len(tpr1) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, tpr1, marker='o', linestyle='-', color='b', label='True Positive Ratio 1')
    plt.plot(iterations, tpr2, marker='s', linestyle='--', color='r', label='True Positive Ratio 2')
    plt.plot(iterations, tpr3, marker='^', linestyle='-.', color='g', label='True Positive Ratio 3')

    plt.xlabel('Iterations')
    plt.ylabel('True Positive Ratio')
    plt.title('True Positive Ratio vs. Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig("true_positive_ratio.pdf")
    plt.show()




def perform_cross_validation(train_dataset: Dataset, test_dataset: Dataset, settings: Settings, k=5, random_seed=None, patience=3):
    """
    Perform K-Fold cross-validation on the dataset.

    :param test_dataset: Test dataset.
    :param train_dataset: Features dataset.
    :type train_dataset: numpy.ndarray
    :param y: Target labels.
    :type y: numpy.ndarray
    :param k: Number of folds for cross-validation.
    :type k: int
    :return: List of accuracy scores for each fold.
    :rtype: list of float
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)

    graphs_list = train_dataset.get_graphs_list()
    test_errors_cross_validation_list = []

    '''
    # create the different dataset over which we train the k folds
    list_train_dataset_crossvalidation = []
    list_test_dataset_crossvalidation = []
    for train_index, test_index in kf.split(graphs_list):
        list_train_dataset_crossvalidation.append(Dataset([graphs_list[i] for i in train_index]))
        list_test_dataset_crossvalidation.append(Dataset([graphs_list[i] for i in test_index]))

    # -----------------------------------------------------------------------------------------------------------------
    # Parallelization
    input_for_parallelization = list(zip(list_train_dataset_crossvalidation, list_test_dataset_crossvalidation))
    with mp.Pool(min(Settings.max_number_of_cores, len(Settings.considered_metal_centers))) as pool:
        array_of_outputs = pool.map(parallelize_cross_validation, input_for_parallelization)
    for model in array_of_outputs:
        test_errors_cross_validation_list.append(model.test_errors)
    # -----------------------------------------------------------------------------------------------------------------
    '''
    # old for loop, it works, but it is not parallelized
    for train_index, test_index in kf.split(graphs_list):

        train_dataset_crossvalidation = Dataset([graphs_list[i] for i in train_index])
        test_dataset_crossvalidation = Dataset([graphs_list[i] for i in test_index])

        if Settings.wrapper_boosting is False:
            model = PatternBoosting(settings = settings)
            model.training(train_dataset_crossvalidation, test_dataset_crossvalidation)
            test_errors_cross_validation_list.append(model.test_error)
        else:

            model = WrapperPatternBoosting(settings = settings)
            model.train(train_dataset_crossvalidation, test_dataset_crossvalidation)
            test_errors_cross_validation_list.append(model.get_wrapper_test_error())

    test_errors_cross_validation_list = np.array(test_errors_cross_validation_list)
    test_error_sum = np.sum(test_errors_cross_validation_list, axis=0)
    overfitting_iteration = 0

    overfitting_iteration = early_stopping(test_errors=test_error_sum, patience=patience)

    if overfitting_iteration <= 2:
        return None, None, None, None

    # run the algorithm training over th whole train dataset and see the error in the test dataset
    if Settings.wrapper_boosting is False:
        model = PatternBoosting(settings = settings)
    else:
        model = WrapperPatternBoosting(settings=settings)
    model.settings.maximum_number_of_steps = overfitting_iteration

    if Settings.wrapper_boosting is False:
        model.training(train_dataset, test_dataset)
        test_error = model.test_error
        # get the number of selected_paths
        n_selected_paths = len(model.get_selected_paths_in_boosting_matrix())
    else:
        model.train(train_dataset, test_dataset)
        test_error = model.get_wrapper_test_error()
        # get the number of selected_paths
        n_selected_paths = len(model.get_selected_paths())

    return overfitting_iteration, test_error, n_selected_paths, test_errors_cross_validation_list


def print_dict_sorted_by_values(d: dict):
    """
    Prints the dictionary sorted by its values.

    :param d: Dictionary to be sorted and printed
    :type d: dict
    """
    # Sort the dictionary by its values
    sorted_items = sorted(d.items(), key=lambda item: item[1])

    # Print the sorted dictionary
    for key, value in sorted_items:
        print(f"{key}: {value}")


def plot_label_distribution(label_counts, save_fig=True):
    # Sort the labels based on their tuple values
    sorted_labels = sorted(label_counts.keys())

    # Convert tuple labels to atomic names and get counts
    labels_str = [periodictable.elements[label[0]].symbol for label in sorted_labels]
    counts = [label_counts[label] for label in sorted_labels]

    # Normalize counts for colormap
    norm = plt.Normalize(min(counts), max(counts))
    colors = cm.viridis(norm(counts))

    plt.figure(figsize=(12, 7))

    # Create a bar plot with the colors representing heights
    sns.barplot(x=labels_str, y=counts, palette=colors)

    plt.xlabel('Element')
    plt.ylabel('Counts')
    plt.title('Distribution of Elements in the Dataset')
    plt.xticks(rotation=90)  # Rotate labels for better readability
    if save_fig is True:
        plt.savefig("label_distribution.pdf")
    plt.show()
    plt.close()


def plot_signal_to_noise_ratio(average_y_value, noise_variance_list, variance_errors, mean_errors, min_errors,
                               max_errors, save_fig=True, name_fig="signal_to_noise_ratio.pdf"):
    # Assuming we're plotting these errors against a sequential index or a list of parameters `x`
    # x_values = np.divide(np.square(average_y_value), np.array(noise_variance_list))
    x_values = noise_variance_list

    # Create a sorted index based on x_values
    sorted_indices = np.argsort(x_values)
    x_values = np.array(x_values)[sorted_indices]
    mean_errors = np.array(mean_errors)[sorted_indices]
    min_errors = np.array(min_errors)[sorted_indices]
    max_errors = np.array(max_errors)[sorted_indices]
    variance_errors = np.array(variance_errors)[sorted_indices]

    # Convert variance to standard deviation for error bars
    std_errors = np.sqrt(variance_errors)

    # Plotting the graph
    plt.figure(figsize=(10, 6))
    # plt.errorbar(x_values, mean_errors, yerr=std_errors, fmt='o', label='Mean Error ± Std Deviation', ecolor='red', elinewidth=3, capsize=0)
    plt.errorbar(x_values, mean_errors, fmt='o', label='Mean Error', ecolor='red',
                 elinewidth=3, capsize=0)
    plt.fill_between(x_values, min_errors, max_errors, color='b', alpha=0.2, label='Min-Max Range')

    plt.title('Performances with increase noise')
    plt.xlabel('Added noise variance')
    plt.ylabel('Test MSE')
    # Round x_values for cleaner display (optional)
    rounded_x_values = [round(x, 1) for x in x_values]

    # Create a smaller subset of x-ticks
    # For example, you can select every other tick or a specific number at regular intervals
    tick_indices = range(0, len(x_values), 1)  # Every other index
    selected_ticks = [rounded_x_values[i] for i in tick_indices]
    selected_tick_positions = [x_values[i] for i in tick_indices]
    # Apply the selected ticks
    plt.xticks(ticks=selected_tick_positions, labels=selected_ticks,
               rotation=45)  # Rotate the x-tick labels to prevent overlap

    plt.legend()

    plt.grid(True)
    if save_fig is True:
        plt.savefig(name_fig)
    plt.show()


def save_data(data, names, directory, create_unique_subfolder=False):
    for i, variable in enumerate(data):
        data_reader.save_data(variable,
                              filename=names[i],
                              directory=directory,
                              create_unique_subfolder=create_unique_subfolder)


def plot_test_error_vs_iterations(test_errors_per_iteration: list[list[float]], save_fig=True,
                                  name_fig="test_error_with_different_lengths.pdf"):
    """
    Plots the test error with standard deviation error bars, max, min, and uses a color gradient
    to indicate the number of series contributing to each data point. Includes a legend for standard deviation.
    """
    # Find the maximum length of the test error series
    max_length = max(len(single_run) for single_run in test_errors_per_iteration)

    # Pad shorter series with np.nan and count valid (non-nan) entries for each iteration
    padded_test_errors = np.full((len(test_errors_per_iteration), max_length), np.nan)
    valid_counts = np.zeros(max_length)
    for i, single_run in enumerate(test_errors_per_iteration):
        length = len(single_run)
        padded_test_errors[i, :length] = single_run
        valid_counts[:length] += 1

    # Calculate statistics using functions that ignore np.nan
    mean_test_error = np.nanmean(padded_test_errors, axis=0)
    std_test_error = np.nanstd(padded_test_errors, axis=0)

    # Generate a colormap based on the valid_counts
    color_map = cm.viridis(valid_counts / max(valid_counts))

    iterations = np.arange(1, max_length + 1)

    plt.figure(figsize=(10, 6))

    # Error bar plot with gradient color coding
    for i, (mean, std, color) in enumerate(zip(mean_test_error, std_test_error, color_map)):
        if i == 0:
            plt.errorbar(i + 1, mean, yerr=std, fmt='-o', markersize=4, color=color, ecolor='lightgrey', elinewidth=3,
                         capsize=2, label='Std. Dev.')
        plt.errorbar(i + 1, mean, yerr=std, fmt='-o', markersize=4, color=color, ecolor='lightgrey', elinewidth=3)

    # Create a color bar to indicate the number of series contributing
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=mcolors.Normalize(vmin=0, vmax=max(valid_counts)))
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='vertical')
    cbar.set_label('Number of Contributing Series')

    plt.xlabel('Iterations')
    plt.ylabel('Test Error MSE')
    plt.title('Test Error')
    plt.legend(loc='best')  # Adjust the legend location as needed
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    if save_fig:
        plt.savefig(name_fig)
    plt.show()


def cross_validation_synthetic_dataset(folder_relative_path, n_iterations, k_folds, settings: Settings):
    directory = data_reader.get_save_location(folder_relative_path=folder_relative_path, unique_subfolder=False)

    # check if other simulations have been done if so, load them

    name_addition = str(settings.noise_variance) + '_scenario_' + str(settings.synthetic_dataset_scenario)

    try:
        list_overfitting_iterations = data_reader.load_data(directory=directory,
                                                            filename="list_overfitting_iterations_" + name_addition)
    except:
        list_overfitting_iterations = []
    try:
        list_of_test_errors = data_reader.load_data(directory=directory,
                                                    filename="list_of_test_errors_" + name_addition)
    except:
        list_of_test_errors = []
    try:
        list_n_selected_paths = data_reader.load_data(directory=directory,
                                                      filename="list_n_selected_paths_" + name_addition)
    except:
        list_n_selected_paths = []

    # list_overfitting_iterations = []
    # list_of_test_errors = []
    # list_n_selected_paths = []

    # Seed and retrieve the values

    random_generator = random.Random()
    random_generator.seed(settings.random_split_test_dataset_seed + 2)
    n_min = 0
    n_max = 20000000

    i = 0
    while i < n_iterations:
        print("iteration number ", i)

        dataset = load_dataset(settings= settings)
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, settings.test_size,
                                                                          random_split_seed=settings.random_split_test_dataset_seed)

        overfitting_iteration, test_error, n_selected_paths, _ = perform_cross_validation(train_dataset, test_dataset,
                                                                                          settings=settings,
                                                                                          k=k_folds,
                                                                                          random_seed=random_generator.randint(
                                                                                              n_min,
                                                                                              n_max))
        if overfitting_iteration is None:
            continue
        i += 1
        list_overfitting_iterations.append(overfitting_iteration)
        list_of_test_errors.append(test_error)
        list_n_selected_paths.append(n_selected_paths)

        # just to save every 10 iteration in order to have a backup
        if i % 10 == 0:
            name_addition = str(settings.noise_variance) + '_scenario_' + str(settings.synthetic_dataset_scenario)
            save_data(data=[list_overfitting_iterations, list_of_test_errors, list_n_selected_paths],
                      names=['list_overfitting_iterations_' + name_addition, 'list_of_test_errors_' + name_addition,
                             'list_n_selected_paths_' + name_addition], directory=directory)

    directory = data_reader.get_save_location(folder_relative_path=folder_relative_path, unique_subfolder=False)
    name_addition = str(settings.noise_variance) + '_scenario_' + str(settings.synthetic_dataset_scenario)
    save_data(data=[list_overfitting_iterations, list_of_test_errors, list_n_selected_paths],
              names=['list_overfitting_iterations_' + name_addition, 'list_of_test_errors_' + name_addition,
                     'list_n_selected_paths_' + name_addition], directory=directory)

    return list_overfitting_iterations, list_of_test_errors, list_n_selected_paths


def plot_patience_overfitting_evolution(overfitting_evolution, patience_range, saving_location):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(patience_range, overfitting_evolution, marker='o')

    ax.set_title('Overfitting Iteration vs Patience')
    ax.set_xlabel('Patience')
    ax.set_ylabel('Overfitting Iteration')
    ax.grid(True)

    # Specify the filename and path to save the plot
    plot_filename = "overfitting_vs_patience.png"
    plot_path = os.path.join(saving_location, plot_filename)

    plt.savefig(plot_path)

    plt.show()
