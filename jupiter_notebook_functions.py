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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
from classes.pattern_boosting import PatternBoosting
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
import pandas as pd
import copy
import matplotlib.pyplot as plt
from classes.analysis import Analysis
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



def get_XGB_error_and_variable_importance(max_path_length, frequency_matrix, labels, max_number_of_learners,
                                          xgb_settings=None):
    header = list(frequency_matrix.columns)

    columns_to_be_removed = []
    print("Setting up matrix ready")
    for i, path in enumerate(header):
        if len(path) > max_path_length:
            columns_to_be_removed.append(path)
    print("Frequency matrix ready")
    x_train, x_test, y_train, y_test = train_test_split(frequency_matrix.drop(columns=columns_to_be_removed),
                                                        labels,
                                                        test_size=Settings.test_size,
                                                        random_state=Settings.random_split_test_dataset_seed)

    if xgb_settings is None:
        xgb_settings = Settings.xgb_model_parameters

    # for i in range(1, max_number_of_learners + 1,10):

    if isinstance(max_number_of_learners, Iterable):
            learners_numbers=max_number_of_learners
    else:
        learners_numbers = list(range(1, max_number_of_learners, 1))
    xgb_test_err = []
    xgb_train_err = []
    features_importance = None
    for i in learners_numbers:
        print("Learner number: ", i)
        # create xgb model
        # print("number of base learners: ", i)
        xgb_settings['n_estimators'] = i - 1
        xgb_model = XGBRegressor(**xgb_settings)

        xgb_model.fit(x_train, y_train)

        # test the model
        y_test_pred = xgb_model.predict(x_test)
        y_train_pred = xgb_model.predict(x_train)

        if Settings.final_evaluation_error == "MSE":
            model_test_error = metrics.mean_squared_error(y_test, y_test_pred)
            model_train_error = metrics.mean_squared_error(y_train, y_train_pred)
        elif Settings.final_evaluation_error == "absolute_mean_error":
            model_test_error = metrics.mean_absolute_error(y_test, y_test_pred)
            model_train_error = metrics.mean_absolute_error(y_train, y_train_pred)
        else:
            raise ValueError("measure error not found")
        xgb_test_err.append( model_test_error)
        xgb_train_err.append( model_train_error)
        features_importance = xgb_model.feature_importances_

    # plot feature importance
    print("max path length: ", max_path_length)
    plot_importance(xgb_model)
    print("max path length: ", max_path_length)
    plt.show()
    print("max path length: ", max_path_length)
    xgb_test_err = np.array(xgb_test_err)
    return xgb_test_err, xgb_train_err, features_importance


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
    ax.set_ylim(0.00075, 0.00125)
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


def plot_graphs(x, y, tittle: str, x_label: str = "", y_label: str = "", show=True, save=True, y2=None, x2=None):
    plt.style.use('ggplot')

    fig, ax = plt.subplots()
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
