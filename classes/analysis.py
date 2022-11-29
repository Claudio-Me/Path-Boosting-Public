from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from settings import Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pathlib
import os
import sys


class Analysis:

    def plot_informations(self, x, y, tittle: str, x_label: str = "", y_label: str = ""):

        plt.style.use('ggplot')

        fig, ax = plt.subplots()

        # Using set_dashes() to modify dashing of an existing line

        if len(x) > Settings.tail:
            ax.plot(x[-Settings.tail:], y[-Settings.tail:], label='')
        else:
            ax.plot(x, y, label='')

        ax.set_title(tittle)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

        # plt.grid()

        # plot only integers on the x axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        saving_location = self.__get_save_location(tittle, '.png')

        plt.savefig(saving_location)
        plt.show()

    @staticmethod
    def __get_save_location(file_name: str, file_extension: str) -> str:
        # it createst a new folder and returns the location of a graph file with name tittle
        location = Settings.graphs_folder
        if file_extension[0] != '.':
            raise TypeError("File extension must start with a dot")

        if location[-1] != '/':
            location = location + '/'

        if Settings.algorithm == "R":
            folder_name = "R_" + str(
                Settings.maximum_number_of_steps) + '_' + Settings.r_base_learner_name + '_' + Settings.family
        elif Settings.algorithm == "Full_xgb":
            folder_name = "Xgb_" + str(Settings.maximum_number_of_steps)
        elif Settings.algorithm == "Xgb_step":
            folder_name = "Xgb_step_" + str(Settings.maximum_number_of_steps)
        else:
            raise TypeError("Selected algorithm not recognized")

        if not os.path.exists(location + folder_name):
            os.makedirs(location + folder_name)
        print(location + folder_name)
        folder_name = folder_name + "/"

        if Settings.maximum_number_of_steps <= Settings.tail:
            file_name = file_name.replace(" ", "_") + file_extension
        else:
            file_name = file_name.replace(" ", "_") + "_tail_" + str(Settings.tail) + file_extension
        return location + folder_name + file_name

    def print_boosting_matrix_information(self, boosting_matrix: BoostingMatrix):
        print(boosting_matrix)
        saving_location = self.__get_save_location("boosting_matrix_info", '.txt')

        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(saving_location, 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(boosting_matrix)
            sys.stdout = original_stdout  # Reset the standard output to its original value

    def print_test_dataset_info(self, dataset):
        average_test_label = np.mean(dataset.labels)
        print("average value for test label: ", average_test_label)

    def analyse_path_length_distribution(self, boosting_matrix: BoostingMatrix):
        self.__plot_histogram_of_path_length(boosting_matrix)
        self.__plot_histogram_of_path_length_importance(boosting_matrix)

    def __plot_histogram_of_path_length(self, boosting_matrix: BoostingMatrix):
        tittle = "Histogram of path length"
        path_length_vector = np.zeros(len(boosting_matrix.header))

        for i in range(len(boosting_matrix.header)):
            path_length_vector[i] = len(boosting_matrix.header[i])

        max_path_length = int(np.amax(path_length_vector))

        plt.style.use('ggplot')

        fig, ax = plt.subplots()

        ax.hist(path_length_vector, bins=max_path_length, ec="k")

        # plot only integers on the x axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # to have only integers in the y axis
        ax.locator_params(axis='y', integer=True)

        ax.set_xlabel('Path length')
        ax.set_title(tittle)

        saving_location = self.__get_save_location(tittle, '.png')

        plt.savefig(saving_location)

        plt.show()

    def __plot_histogram_of_path_length_importance(self, boosting_matrix: BoostingMatrix):
        tittle = "Path length importance"
        path_length_vector = np.zeros(len(boosting_matrix.header))

        for i in range(len(boosting_matrix.header)):
            path_length_vector[i] = len(boosting_matrix.header[i])

        max_path_length = int(np.amax(path_length_vector))

        length_importance = [0] * max_path_length
        for i in range(len(boosting_matrix.header)):
            path_length = len(boosting_matrix.header[i])
            length_importance[path_length - 1] += boosting_matrix.patterns_importance[i]

        # normalize the importance vector
        length_importance = np.array(length_importance)
        length_importance *= 1 / length_importance.max()

        plt.style.use('ggplot')

        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.bar(range(1, max_path_length + 1), length_importance)

        ax.set_xlabel('Path length')
        ax.set_ylabel('Importance')
        ax.set_title(tittle)

        saving_location = self.__get_save_location(tittle, '.png')

        plt.savefig(saving_location)

        plt.show()
