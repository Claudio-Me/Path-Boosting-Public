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

from xgboost import plot_importance

from sklearn import metrics
import numpy as np
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
import matplotlib.pyplot as plt
from classes.analysis_patternboosting import AnalysisPatternBoosting
from data.load_dataset import load_dataset
from data import data_reader
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from multiprocessing.dummy import Pool as ThreadPool
import functools
import copy
from data import data_reader

import pathlib
import os
import sys
from collections.abc import Iterable
from matplotlib.ticker import MaxNLocator
from typing import List


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
        x_test = pattern_boosting.create_boosting_matrix_for(pattern_boosting.test_dataset)
    elif isinstance(pattern_boosting, WrapperPatternBoosting):
        x_test = pattern_boosting.create_ordered_boosting_matrix(pattern_boosting.test_dataset)
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


import matplotlib.pyplot as plt
