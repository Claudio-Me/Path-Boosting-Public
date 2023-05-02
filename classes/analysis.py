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

import pathlib
import os
import sys


class Analysis:
    def __init__(self, train_predictions=None, test_predictions=None, train_error=None, test_error=None):
        self.train_predictions = train_predictions
        self.test_predictions = test_predictions

        self.train_error = train_error
        self.test_error = test_error

    def load_and_analyze(self, directory, show=True, save=True):

        pattern_boosting = data_reader.load_data(directory=directory, filename="pattern_boosting")
        try:
            synthetic_dataset = data_reader.load_data(directory=directory, filename="synthetic_dataset")
        except:
            synthetic_dataset = None
        self.all_analysis(pattern_boosting, synthetic_dataset, show=show, save=save)

    def all_analysis(self, pattern_boosting: PatternBoosting, synthetic_dataset=None, show=True, save=True):
        self.errors_plots(pattern_boosting, show=show, save=save)
        self.average_path_length_plot(pattern_boosting, show=show, save=save)
        self.print_performance_information(pattern_boosting, pattern_boosting.train_error,
                                           pattern_boosting.test_error,
                                           pattern_boosting.training_dataset, show=show, save=save)
        if synthetic_dataset is not None:
            self.print_synthetic_dataset_performances(synthetic_dataset=synthetic_dataset,
                                                      pattern_boosting=pattern_boosting, show=show, save=save)

    def print_synthetic_dataset_performances(self, synthetic_dataset: SyntheticDataset,
                                             pattern_boosting: PatternBoosting = None, show=True, save=True):
        synthetic_dataset.create_and_save_tables_in_latex(pattern_boosting_model=pattern_boosting, show=show, save=save)

    def average_path_length_plot(self, pattern_boosting: PatternBoosting, show=True, save=True):
        '''
        It plots all the graphs relative to the path length
        '''
        self.plot_graphs(pattern_boosting.number_of_learners, pattern_boosting.average_path_length,
                         tittle="Average path length",
                         x_label="number of learners", y_label="average path length", show=show, save=save)

        self.analyse_path_length_distribution(pattern_boosting.boosting_matrix, show=show, save=save)

        if pattern_boosting.test_dataset is not None:
            # self.analysis.plot_labels_histogram(self.training_dataset.labels, self.test_dataset.labels, tittle="Real labels")
            # self.analysis.plot_labels_histogram(self.predict(training_dataset), self.predict(test_dataset), tittle="Predicted Labels")
            if self.test_predictions is None:
                self.test_predictions = pattern_boosting.predict(pattern_boosting.test_dataset)
            self.plot_labels_histogram(pattern_boosting.test_dataset.labels,
                                       self.test_predictions,
                                       tittle="Predicted vs real y", legend1="real", legend2="predicted", show=show,
                                       save=save)

    def errors_plots(self, pattern_boosting: PatternBoosting, show=True, save=True):

        self.train_error = pattern_boosting.train_error
        self.test_error = pattern_boosting.test_error

        # cut first n points
        cut_point = 0

        if Settings.final_evaluation_error == "MSE":
            y_label = "MSE"
        elif Settings.final_evaluation_error == "absolute_mean_error":
            y_label = "abs mean err"
        else:
            raise ValueError("measure error not found")
        if Settings.estimation_type == EstimationType.regression:
            self.plot_graphs(pattern_boosting.number_of_learners[cut_point:],
                             pattern_boosting.train_error[cut_point:],
                             tittle="train error",
                             x_label="number of learners",
                             y_label=y_label,
                             show=show, save=save)

            if pattern_boosting.test_dataset is not None:
                self.plot_graphs(pattern_boosting.number_of_learners[cut_point:],
                                 pattern_boosting.test_error[cut_point:],
                                 tittle="test error",
                                 x_label="number of learners",
                                 y_label=y_label,
                                 show=show, save=save)
        elif Settings.estimation_type == EstimationType.classification:
            self.plot_graphs(pattern_boosting.number_of_learners, pattern_boosting.train_error,
                             tittle="train classification",
                             x_label="number of learners", y_label="jaccard score",
                             show=show, save=save)

            if pattern_boosting.test_dataset is not None:
                self.plot_graphs(pattern_boosting.number_of_learners, pattern_boosting.test_error,
                                 tittle="test classification",
                                 x_label="number of learners", y_label="jaccard score",
                                 show=show, save=save)

        self.__scatterplot_test_prediction_vs_labels(pattern_boosting, show=show, save=save)

    def plot_graphs(self, x, y, tittle: str, x_label: str = "", y_label: str = "", show=True, save=True):

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
        saving_location = data_reader.get_save_location(tittle, '.png')

        if save is True:
            plt.savefig(saving_location)
        if show is True:
            plt.show()

    def print_performance_information(self, pattern_boosting: PatternBoosting, train_error_vector, test_error_vector,
                                      dataset: Dataset, show=True, save=True):

        string = "\n--------------------------------------------------------------------------------\n"
        string += "Training dataset dimension: " + str(pattern_boosting.training_dataset.get_dimension()) + '\n'
        string += "Test dataset dimension: " + str(pattern_boosting.test_dataset.get_dimension()) + '\n'

        string += "Number of iterations: " + str(len(train_error_vector)) + '\n'
        string += "Number of iterations: " + str(len(train_error_vector)) + '\n'
        string += "Boosting Matrix:\n" + str(pattern_boosting.boosting_matrix) + '\n'
        string += "achieved train error: " + str(train_error_vector[-1])
        string += "achieved test error: " + str(test_error_vector[-1])
        string += repr(dataset)
        if show is True:
            print(string)
        if save is True:
            saving_location = data_reader.get_save_location("boosting_matrix_info", '.txt')

            original_stdout = sys.stdout  # Save a reference to the original standard output

            with open(saving_location, 'w') as f:
                sys.stdout = f  # Change the standard output to the file we created.
                string += "Train error vector:\n" + str(train_error_vector) + '\n'
                string += "Test error vector:\n" + str(test_error_vector) + '\n'
                print(string)
                sys.stdout = original_stdout  # Reset the standard output to its original value

    def analyse_path_length_distribution(self, boosting_matrix: BoostingMatrix, show=True, save=True):
        self.__plot_bar_plot_of_path_length(boosting_matrix, show=show, save=save)
        self.__plot_histogram_of_path_length_importance(boosting_matrix, show=show, save=save)

    def plot_labels_histogram(self, train_labels, test_labels=None, tittle=None, legend1="train", legend2="test",
                              show=True, save=True):
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
        saving_location = data_reader.get_save_location(tittle, '.png')

        if save is True:
            plt.savefig(saving_location)
        if show is True:
            plt.show()

    def __scatterplot_test_prediction_vs_labels(self, pattern_boosting: PatternBoosting, show=True, save=True):
        if pattern_boosting.test_dataset is not None:
            if self.test_predictions is None:
                self.test_predictions = pattern_boosting.predict(pattern_boosting.test_dataset)

            tmp_weird_graphs=[]
            for number,prediction in enumerate(self.test_predictions):
                if prediction<0.18:
                    tmp_weird_graphs.append(pattern_boosting.test_dataset.get_graph_number(number))

            boosting_matrix_of_weird_graphs=pattern_boosting.generate_boosting_matrix(Dataset(tmp_weird_graphs))


            plt.style.use('ggplot')

            fig, ax = plt.subplots()
            ax.scatter(self.test_predictions, pattern_boosting.test_dataset.get_labels(), c='black')
            ax.set_xlabel('Predictions')
            ax.set_ylabel('True label')
            max_all = max(list(self.test_predictions) + pattern_boosting.test_dataset.get_labels())
            min_all = min(list(self.test_predictions) + pattern_boosting.test_dataset.get_labels())
            ax.plot([min_all, max_all], [min_all, max_all], color='red')

        if save is True:
            saving_location = data_reader.get_save_location(file_name="prediction_vs_true_value", file_extension='.png')
            plt.savefig(saving_location)
        if show is True:
            plt.show()

    def __plot_bar_plot_of_path_length(self, boosting_matrix: BoostingMatrix, show=True, save=True):
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

        saving_location = data_reader.get_save_location(tittle, '.png')

        if save is True:
            plt.savefig(saving_location)
        if show is True:
            plt.show()

    def __plot_histogram_of_path_length_importance(self, boosting_matrix: BoostingMatrix, show=True, save=True):
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

        saving_location = data_reader.get_save_location(file_name=tittle, file_extension='.png')

        if save is True:
            plt.savefig(saving_location)
        if show is True:
            plt.show()
