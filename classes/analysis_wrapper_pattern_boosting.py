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
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
import pathlib
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


class AnalysisWrapperPatternBoosting:
    def __init__(self, wrapper_pattern_boosting: WrapperPatternBoosting):
        self.wrapper_pattern_boosting = wrapper_pattern_boosting

    def plot_top_n_paths_heatmap(self, n: int| None = None):
        paths, importances = self.wrapper_pattern_boosting.get_patterns_importance()
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

        max_len = max(len(path) for path in top_paths)  # Change here

        # Creating the heatmap matrix where each row corresponds to a top path
        heatmap_matrix = np.zeros((n, max_len))

        # Iterating through each top path and setting the importance for each subpath
        for i, path in enumerate(top_paths):
            for j in range(1, len(path) + 1):
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
        y_labels = [f"Path {'-'.join(map(str, path))}" for path in top_paths]
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(y_labels)

        # Setting the labels for the x-axis to represent the subpath lengths
        ax.set_xticks(np.arange(max_len))
        ax.set_xticklabels(range(1, max_len + 1))

        plt.title(f'Top {n} Paths by Importance Heatmap')
        plt.ylabel('Paths')
        plt.xlabel('Subpath Length')

        plt.show()