from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from settings import Settings
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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

        plt.show()

    def print_matrix_header(self, boosting_matrix: BoostingMatrix):
        print("Boosting matrix header:")
        print(boosting_matrix.header)
        print("Selected Paths")
        print(boosting_matrix.translate_header_to_atom_symbols())

        print("number of explored paths: ", len(boosting_matrix.header))
        print("max path length: ", self.__max_path_length(boosting_matrix))
        print("average path length: ", self.average_path_length(boosting_matrix))

    def print_info(self, test_dataset):
        average_test_label = np.mean(test_dataset.labels)
        print("average value for test label: ", average_test_label)

    @staticmethod
    def average_path_length(boosting_matrix):
        average_path_length = 0

        for path in boosting_matrix.header:
            average_path_length = average_path_length + len(path)
        average_path_length = average_path_length / len(boosting_matrix.header)
        return average_path_length

    def __max_path_length(self, boosting_matrix):
        max_length = 0

        for path in boosting_matrix.header:
            if len(path) > max_length:
                max_length = len(path)
        return max_length
