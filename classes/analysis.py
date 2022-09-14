from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from settings import Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pathlib
import os


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
        saving_location = self.__get_save_location(tittle)

        plt.savefig(saving_location)
        plt.show()

    @staticmethod
    def __get_save_location(tittle: str) -> str:
        # it createst a new folder and returns the location of a graph file with name tittle
        location = Settings.graphs_folder
        if location[-1] != '/':
            location = location + '/'

        if Settings.use_R is True:
            folder_name = "R_" + str(
                Settings.maximum_number_of_steps) + '_' + Settings.r_base_learner_name + '_' + Settings.family
        else:
            folder_name = "Xgb_" + str(Settings.maximum_number_of_steps)

        if not os.path.exists(location + folder_name):
            os.makedirs(location + folder_name)
        print(location + folder_name)
        folder_name = folder_name + "/"

        if Settings.maximum_number_of_steps <= Settings.tail:
            file_name = tittle.replace(" ", "_") + ".png"
        else:
            file_name = tittle.replace(" ", "_") + "_tail_" + str(Settings.tail) + ".png"
        return location + folder_name + file_name

    def print_boosting_matrix_information(self, boosting_matrix: BoostingMatrix):
        print(boosting_matrix)

    def print_test_dataset_info(self, dataset):
        average_test_label = np.mean(dataset.labels)
        print("average value for test label: ", average_test_label)

    def analyse_path_length_distribution(self, boosting_matrix):
        tittle="Histogram of path length"
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

        saving_location = self.__get_save_location(tittle)

        plt.savefig(saving_location)

        plt.show()


