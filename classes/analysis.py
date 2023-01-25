from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from classes.dataset import Dataset
from settings import Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import Counter
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

        # plot only integers on the x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        saving_location = self.get_save_location(tittle, '.png')

        plt.savefig(saving_location)
        plt.show()

    @staticmethod
    def get_save_location(file_name: str, file_extension: str) -> str:
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

        folder_name = folder_name + "_max_path_length_" + str(Settings.max_path_length)
        if not os.path.exists(location + folder_name):
            os.makedirs(location + folder_name)
        folder_name = folder_name + "/"

        if Settings.maximum_number_of_steps <= Settings.tail:
            file_name = file_name.replace(" ", "_") + file_extension
        else:
            file_name = file_name.replace(" ", "_") + "_tail_" + str(Settings.tail) + file_extension
        return location + folder_name + file_name

    def print_performance_information(self, boosting_matrix: BoostingMatrix, train_error_vector, test_error_vector,
                                      dataset: Dataset):
        print("Number of iterations: ", len(train_error_vector))
        print(boosting_matrix)
        train_error_string = "achieved train error: " + str(train_error_vector[-1])
        print(train_error_string)
        test_error_string = "achieved test error: " + str(test_error_vector[-1])
        print(test_error_string)
        print(dataset)

        saving_location = self.get_save_location("boosting_matrix_info", '.txt')

        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(saving_location, 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("Number of iterations: ", Settings.maximum_number_of_steps)
            print("Boosting matrix:")
            print(boosting_matrix)
            print("\n\n")
            print(dataset)
            print("\n")
            print(train_error_string)
            print(str(train_error_vector) + "\n")
            print(test_error_string)
            print(test_error_vector)

            sys.stdout = original_stdout  # Reset the standard output to its original value

    def analyse_path_length_distribution(self, boosting_matrix: BoostingMatrix):
        self.__plot_bar_plot_of_path_length(boosting_matrix)
        self.__plot_histogram_of_path_length_importance(boosting_matrix)

    def plot_labels_histogram(self, train_labels, test_labels=None, tittle=None, legend1="train", legend2="test"):
        kwargs = dict(alpha=0.5, histtype='bar', density=True, stacked=False)
        plt.style.use('ggplot')

        fig, ax = plt.subplots()
        ax.hist((train_labels, test_labels), **kwargs, label=(legend1, legend2), color=('g', 'r'))
        if test_labels is not None:
            pass
            # ax.hist(test_labels,**kwargs,label="test",color= 'r')

        # ax.set_ylabel('Importance')
        ax.set_title(tittle)
        ax.legend()
        saving_location = self.get_save_location(tittle, '.png')

        plt.savefig(saving_location)

        plt.show()

    def __plot_bar_plot_of_path_length(self, boosting_matrix: BoostingMatrix):
        tittle = "Bar plot of path length"
        path_length_vector = np.zeros(len(boosting_matrix.header))

        for i in range(len(boosting_matrix.header)):
            path_length_vector[i] = len(boosting_matrix.header[i])

        max_path_length = int(np.amax(path_length_vector))

        path_length_counter = Counter(path_length_vector)

        path_length = path_length_counter.keys()
        path_length = [str(int(length)) for length in path_length]
        number_of_paths = list(path_length_counter.values())
        number_of_paths = [int(n) for n in number_of_paths]

        plt.style.use('ggplot')

        fig, ax = plt.subplots()

        ax.bar(path_length, number_of_paths, color='maroon')
        # plot only integers on the x axis
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # to have only integers in the y axis
        ax.locator_params(axis='y', integer=True)

        ax.set_xlabel('Path length')
        ax.set_title(tittle)

        saving_location = self.get_save_location(tittle, '.png')

        plt.savefig(saving_location)

        plt.show()

    def __plot_histogram_of_path_length_importance(self, boosting_matrix: BoostingMatrix):
        tittle = "Path length importance"
        path_length_vector = np.zeros(len(boosting_matrix.header))

        for i in range(len(boosting_matrix.header)):
            path_length_vector[i] = len(boosting_matrix.header[i])

        max_path_length = int(np.amax(path_length_vector))

        length_importance = [0] * max_path_length
        paths_importance = np.array(len(boosting_matrix.get_header()))
        index_max = 0
        max_importance = 0
        for i in range(paths_importance):
            path_importance = boosting_matrix.get_path_importance(boosting_matrix.get_header()[i])
            if max_importance < path_importance:
                max_importance = path_importance
                index_max = i

        for i in range(len(boosting_matrix.get_header())):
            if i != index_max:
                path_length = len(boosting_matrix.header[i])
                path_importance = boosting_matrix.get_path_importance(boosting_matrix.get_header()[i])
                length_importance[path_length - 1] += path_importance

        # normalize the importance vector
        length_importance = np.array(length_importance)
        length_importance *= 1 / length_importance.max()

        plt.style.use('ggplot')

        fig, ax = plt.subplots()

        if False:
            ax.set_yscale('log')
            tittle = tittle + " (log scale)"
        # to do: convert range to string
        path_length_labels = [str(i) for i in range(1, max_path_length + 1)]

        cut_point = 1
        ax.bar(path_length_labels, length_importance)

        ax.set_xlabel('Path length')
        ax.set_ylabel('Importance')
        ax.set_title(tittle)

        saving_location = self.get_save_location(tittle, '.png')

        plt.savefig(saving_location)

        plt.show()
