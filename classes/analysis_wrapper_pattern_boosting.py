from sklearn import metrics
import numpy as np
from classes.boosting_matrix import BoostingMatrix
from classes.dataset import Dataset
from settings import Settings
from matplotlib.ticker import MaxNLocator
from collections import Counter
from classes.pattern_boosting import PatternBoosting
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
from data import data_reader
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
import pathlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from classes.analysis import *


class AnalysisWrapperPatternBoosting:
    def __init__(self, wrapper_pattern_boosting: WrapperPatternBoosting, test_predictions: list[float] | None = None,
                 train_predictions: list[float] | None = None, save: bool = False, show: bool = False):
        self.show: bool = show
        self.save: bool = save
        self.wrapper_pattern_boosting = wrapper_pattern_boosting
        self.test_predictions = test_predictions
        self.train_predictions = train_predictions
        # old method, it works, but it is slow
        # self.train_predictions = self.wrapper_pattern_boosting.predict(self.wrapper_pattern_boosting.test_dataset)
        if self.test_predictions is None:
            self.test_predictions = list(self.wrapper_pattern_boosting.predict_test_dataset_parallel())
        if self.train_predictions is None:
            self.train_predictions = self.wrapper_pattern_boosting.predict_train_dataset_parallel()

    def plot_all_analysis(self, n: int | None = None, synthetic_dataset: SyntheticDataset | None = None):
        self.plot_top_n_paths_heatmap(n)
        self.plot_top_importance_paths(n)
        self.plot_top_importance_paths(n, 2)

        self.plot_performance_scatter_plot(dataset='Train')
        self.plot_performance_scatter_plot(dataset='Test')
        if Settings.show_analysis is True or Settings.save_analysis is True:
            plot_error_evolution(self.wrapper_pattern_boosting.train_error, dataset='Train', show=self.show,
                                 save=self.save)
            plot_error_evolution(self.wrapper_pattern_boosting.test_error, dataset='Test', show=self.show,
                                 save=self.save)

        if synthetic_dataset is not None:
            self.synthetic_dataset_spotted_paths(synthetic_dataset)
            self.performances_on_synthetic_dataset(synthetic_dataset, 'Train')
            self.performances_on_synthetic_dataset(synthetic_dataset, 'Test')
            self.missed_paths_correlations(synthetic_dataset)

    def sub_tuple(self, t1: tuple[int], t2: tuple[int]) -> bool:
        """Checks if t1 is a sub-tuple of t2."""
        lt1, lt2 = len(t1), len(t2)
        for i in range(lt2 - lt1 + 1):
            if t1 == t2[i: i + lt1]:
                return True
        return False

    def __find_longest_path_containing(self, top_path: tuple[int], all_paths: list[tuple[int]], max_length: int) -> \
            tuple[int]:
        """Find the longest path that contains the top path."""
        longest_path = top_path
        for path in all_paths:
            if self.sub_tuple(top_path, path) and len(path) > len(longest_path) and len(path) <= max_length:
                longest_path = path
        return longest_path

    def plot_top_importance_paths(self, n: int, min_length: int = 0):
        paths, importances = self.wrapper_pattern_boosting.get_normalized_patterns_importance()

        if n is None:
            n = len(paths)
        if n > len(paths):
            raise ValueError(f"n ({n}) cannot be greater than the number of paths ({len(paths)})")

            # Sort the paths by their importances and select the top `n`
        sorted_pairs = sorted(zip(paths, importances), key=lambda x: x[1], reverse=True)
        top_paths, top_importances = zip(*sorted_pairs)
        top_paths = list(top_paths)
        top_importances = list(top_importances)

        indices_to_remove = []

        # Iterate over top_paths_extended and collect the indices of duplicates
        for i, path in enumerate(top_paths):
            if len(path) < min_length:
                indices_to_remove.append(i)

        # Go through the indices in reverse order so we don't mess up the
        # subsequent indices after removing an element from the list.
        for index in sorted(indices_to_remove, reverse=True):
            del top_paths[index]
            del top_importances[index]

        top_paths = top_paths[:n]
        top_importances = top_importances[:n]

        # Create a figure and a single subplot
        fig, ax = plt.subplots()

        top_paths=[str(path)for path in top_paths]
        # Create the horizontal bar plot
        ax.barh(range(len(top_paths)), top_importances, tick_label=top_paths)

        # Configure the plot (optional)
        ax.set_xlabel('Importance')
        ax.set_title('Horizontal Bar Plot of Categories and Their Importance')

        # Adjust the layout to prevent cropping
        fig.tight_layout()

        # Display the plot
        if self.show is True:
            plt.show()
        if self.save is True:
            saving_location = data_reader.get_save_location(file_name='paths_importance_bar_plot_'+ str(min_length),
                                                            file_extension=".pdf",
                                                            folder_relative_path='results', unique_subfolder=True)

            fig.savefig(saving_location, format="pdf")

    def plot_top_n_paths_heatmap(self, n: int | None = None):
        paths, importances = self.wrapper_pattern_boosting.get_normalized_patterns_importance()

        if n is None:
            n = len(paths)
        if n > len(paths):
            raise ValueError(f"n ({n}) cannot be greater than the number of paths ({len(paths)})")

        """
        Generates a heatmap for the `n` most important paths.

        This function takes a list of paths and corresponding importance,
        and generates a heatmap for visual representation.

        Each path is represented by a tuple of integers, and corresponds to a row in the heatmap.
        Importance of a path is represented by color intensity on the heatmap. Ordering of elements
        within a path tuple is maintained from left to right on the heatmap.

        Parameters: (No parameters needed, but the firs line of code gets paths and importance)
        :param paths: A list of tuples, each tuple represents a path by a series of integers.
        :type paths: List[Tuple[int]]
        :param importances: A list of importance values corresponding to each path. Importance is represented with a float value.
        :type importances: List[float]
        :param n: The number of top paths by importance to be displayed.
        :type n: int
        :raises ValueError: If 'n' is greater than the number of provided paths.
        :return: None

        Example Usage:
        >>> paths = [(1, 2, 4), (2, 6), (1, 3, 5, 7), (4, 2, 8), (9,)]
        >>> importances = [0.3, 0.7, 0.5, 0.9, 0.2]
        >>> n = 3
        >>> plot_top_n_paths_heatmap(paths, importances, n)
        """
        if n is None:
            n = len(paths)
        if n > len(paths):
            raise ValueError(f"n ({n}) cannot be greater than the number of paths ({len(paths)})")

        # Sort the paths by their importances and select the top `n`
        sorted_pairs = sorted(zip(paths, importances), key=lambda x: x[1], reverse=True)
        top_paths, top_importances = zip(*sorted_pairs[:n])
        top_paths = list(top_paths)
        top_importances = list(top_importances)
        max_path_length = 5
        top_paths_extended = [self.__find_longest_path_containing(top_path, paths, max_length=max_path_length) for
                              top_path in top_paths]

        seen = {}
        indices_to_remove = []
        paths_too_long = []
        importance_too_long = []
        max_path_length = 5

        # Iterate over top_paths_extended and collect the indices of duplicates
        for i, path in enumerate(top_paths_extended):
            if len(path) > max_path_length:
                indices_to_remove.append(i)
                paths_too_long.append(path)
                importance_too_long.append(top_importances[i])
            else:
                if path in seen:
                    indices_to_remove.append(i)
                else:
                    seen[path] = True

        print("top paths that have been excluded because too long:")

        print(paths_too_long)
        print("importance:")
        print(importance_too_long)

        # Go through the indices in reverse order so we don't mess up the
        # subsequent indices after removing an element from the list.
        for index in sorted(indices_to_remove, reverse=True):
            del (top_paths[index])
            del (top_paths_extended[index])
            del top_importances[index]

        max_len = max(len(path) for path in top_paths_extended)

        # Creating the heatmap matrix where each row corresponds to a top path
        heatmap_matrix = np.zeros((len(top_paths), max_len))

        # Iterating through each top path and setting the importance for each subpath
        for i, path in enumerate(top_paths_extended):
            for j in range(2, len(path) + 1):
                # Get the index of the subpath in the original paths list
                subpath = path[:j]
                if subpath in paths:
                    subpath_index = paths.index(subpath)
                    # Use the importance value associated with the subpath
                    heatmap_matrix[i, j - 1] = importances[subpath_index]

        fig, ax = plt.subplots(figsize=(12, 8))
        c = ax.imshow(heatmap_matrix, cmap='viridis', aspect='auto')
        fig.colorbar(c, ax=ax)

        # Setting the labels for the y-axis to the paths (top `n` only)
        y_labels = [
            f"Path {'-'.join(map(str, path))}" + "|" + f"{'-'.join(map(str, top_paths_extended[i][len(path):]))}" for
            i, path in enumerate(top_paths)]
        ax.set_yticks(np.arange(len(top_paths)))
        ax.set_yticklabels(y_labels)

        # Setting the labels for the x-axis to represent the subpath lengths
        ax.set_xticks(np.arange(max_len))
        ax.set_xticklabels(range(1, max_len + 1))
        ax.set_xlim(0.5, max_len - 0.5)

        plt.title(f'Top {len(top_paths)} Paths by Importance Heatmap')
        plt.ylabel('Paths')
        plt.xlabel('Subpath Length')

        # Display the plot
        if self.show is True:
            plt.show()
        if self.save is True:
            saving_location = data_reader.get_save_location(file_name='heat_map',
                                                            file_extension=".pdf",
                                                            folder_relative_path='results', unique_subfolder=True)

            fig.savefig(saving_location, format="pdf")

    def plot_performance_scatter_plot(self, dataset: str):
        if dataset == 'test' or dataset == 'Test':
            predicted_values = self.test_predictions
            actual_values = self.wrapper_pattern_boosting.test_dataset.get_labels()
        elif dataset == 'train' or dataset == 'Train':
            predicted_values = self.train_predictions
            actual_values = self.wrapper_pattern_boosting.train_dataset.get_labels()
        """
        Plot the performance of an algorithm by comparing predicted values to actual values using a scatter plot with color mapping.

        Parameters:
            - predicted_values: A list of predicted float values.
            - actual_values: A list of actual float values.
        """

        # Check if the input lists are of the same length
        if len(predicted_values) != len(actual_values):
            print(dataset)
            print("Error: The lengths of the input lists do not match.")
            return

        # Create a range of indices for plotting
        indices = range(len(predicted_values))

        # Calculate the absolute errors
        errors = np.abs(np.array(predicted_values) - np.array(actual_values))

        # Set up the figure and axes
        fig, ax = plt.subplots()

        # Create a scatter plot with color mapping based on errors
        sc = ax.scatter(actual_values, predicted_values, c=errors, cmap='viridis', marker='o')

        # Determine the range of your data to plot the ideal line
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))

        # Plot ideal line (y=x) on the axis
        ax.plot([min_val, max_val], [min_val, max_val], color="green", linestyle="--", label="Ideal")

        # Set the labels and title
        ax.set_xlabel("Actual Value")
        ax.set_ylabel("Predicted Value")
        ax.set_title("Performance of the Algorithm on " + dataset + " error")

        # Add a color bar
        cbar = fig.colorbar(sc)
        cbar.ax.set_ylabel("Absolute Error")

        # Display the plot
        if self.show is True:
            plt.show()
        if self.save is True:
            saving_location = data_reader.get_save_location(file_name=dataset + '_dataset_scatterplot',
                                                            file_extension=".pdf",
                                                            folder_relative_path='results', unique_subfolder=True)

            fig.savefig(saving_location, format="pdf")

    def synthetic_dataset_spotted_paths(self, synthetic_dataset: SyntheticDataset) -> pd.DataFrame:

        missed_paths = []
        boosting_matrix_header = self.wrapper_pattern_boosting.get_boosting_matrix_header()
        selected_paths = self.wrapper_pattern_boosting.get_selected_paths()
        counter_spotted = 0
        counter_header = 0
        for path in synthetic_dataset.target_paths:
            if path in selected_paths:
                counter_spotted += 1
            else:
                missed_paths.append(path)
            if path in boosting_matrix_header:
                counter_header += 1
        n_spotted_paths = counter_spotted

        print("Total number of target paths: ", len(synthetic_dataset.target_paths))
        oracle_train_error = synthetic_dataset.oracle_model_evaluate(
            graphs_list=self.wrapper_pattern_boosting.train_dataset.get_graphs_list(),
            labels=self.wrapper_pattern_boosting.train_dataset.get_labels())

        data = {
            "Number of target paths": [len(synthetic_dataset.target_paths)],
            "Target paths spotted": [n_spotted_paths],
            "selected paths": [len(selected_paths)],
            "train err": [self.wrapper_pattern_boosting.train_error[-1]],
            "oracle train err": [oracle_train_error]
        }
        if self.wrapper_pattern_boosting.test_error is not None:
            oracle_test_error = synthetic_dataset.oracle_model_evaluate(
                graphs_list=self.wrapper_pattern_boosting.test_dataset.get_graphs_list(),
                labels=self.wrapper_pattern_boosting.test_dataset.get_labels())
            data["test err"] = [self.wrapper_pattern_boosting.test_error[-1]]
            data["oracle test err"] = [oracle_test_error]
        data = pd.DataFrame(data)
        if self.show is True:
            print(data.to_markdown())
        if self.save is True:
            saving_location = data_reader.get_save_location(file_name="synthetic_dataset_spotted_paths_data",
                                                            file_extension=".tex",
                                                            folder_relative_path='results', unique_subfolder=True)
            with open(saving_location, 'w') as tf:
                tf.write(data.style.to_latex())
        return data

    def performances_on_synthetic_dataset(self, synthetic_dataset: SyntheticDataset, dataset: str):
        if dataset == 'test' or dataset == 'Test':
            graphs_list = self.wrapper_pattern_boosting.test_dataset.get_graphs_list()
            wrapper_boosting_preds = self.test_predictions
            labels = self.wrapper_pattern_boosting.test_dataset.get_labels()
        else:
            graphs_list = self.wrapper_pattern_boosting.train_dataset.get_graphs_list()
            wrapper_boosting_preds = self.train_predictions
            labels = self.wrapper_pattern_boosting.train_dataset.get_labels()

        oracle_model_predictions = synthetic_dataset.oracle_model_predict(graphs_list)
        compare_performances_on_synthetic_dataset(test_model_preds=wrapper_boosting_preds,
                                                  oracle_model_preds=oracle_model_predictions,
                                                  true_values=labels,
                                                  dataset=dataset,
                                                  show=self.show,
                                                  save=self.save)

    @staticmethod
    def __highest_corr(df, col_names):
        highest_correlations = {}
        corr_matrix = df.corr()
        for col_name in col_names:
            if col_name in corr_matrix.columns:
                corr_values = corr_matrix[col_name]
                corr_values = corr_values[corr_values.index != col_name]  # exclude self-correlation
                absolute_corr_values = corr_values.abs()  # consider absolute correlation
                highest_corr_coef = max(absolute_corr_values)
                highest_correlations[str(col_name)] = highest_corr_coef
        return highest_correlations

    def missed_paths_correlations(self, synthetic_dataset: SyntheticDataset) -> pd.DataFrame:
        missed_paths = []
        selected_paths = self.wrapper_pattern_boosting.get_selected_paths()
        for path in synthetic_dataset.target_paths:
            if path not in selected_paths:
                missed_paths.append(path)

        highest_correlations_missed_paths = self.wrapper_pattern_boosting.get_train_correlations(missed_paths)
        if len(highest_correlations_missed_paths.keys()) != missed_paths:
            print("Some target paths are not present in training dataset")
        number_of_times_path_is_present = {}
        for path in highest_correlations_missed_paths.keys():
            times_is_present = 0
            for model in self.wrapper_pattern_boosting.pattern_boosting_models_list:
                if model.trained is True:
                    column = model.boosting_matrix.get_path_column(path)
                    if column is None:
                        pass
                    else:
                        times_present_in_model = np.count_nonzero(model.get_boosting_matrix().get_matrix()[:, column])
                        times_is_present = times_is_present + times_present_in_model
            number_of_times_path_is_present[path] = [highest_correlations_missed_paths[path], times_is_present]

        # just change the key from tuple to int
        dataframe_to_save = {}
        for key in number_of_times_path_is_present.keys():
            dataframe_to_save[str(key)] = number_of_times_path_is_present[key]
        dataframe_to_save = pd.DataFrame(dataframe_to_save)
        dataframe_to_save = dataframe_to_save.sort_values(0, axis=1, ascending=False)
        dataframe_to_save.index = ["correlation", "times present"]

        highest_correlations_missed_paths = pd.DataFrame(number_of_times_path_is_present)
        # sort by correlation value
        highest_correlations_missed_paths = highest_correlations_missed_paths.sort_values(0, axis=1, ascending=False)
        highest_correlations_missed_paths.index = ["correlation", "times present"]

        if self.show is True:
            print(dataframe_to_save.iloc[:, : 9].to_markdown())
        if self.save is True:
            saving_location = data_reader.get_save_location(file_name="missed_paths:correlation",
                                                            file_extension=".tex",
                                                            folder_relative_path='results', unique_subfolder=True)
            with open(saving_location, 'w') as tf:
                tf.write(dataframe_to_save.style.to_latex())
        return highest_correlations_missed_paths
