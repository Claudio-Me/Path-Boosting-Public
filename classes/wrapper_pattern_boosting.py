# Takes as input a lyst of parrent boosting models and "merge" them into one
import copy
import functools

import multiprocessing as mp

import warnings
from collections import defaultdict
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from typing import Self
from sklearn import metrics

from classes.boosting_matrix import BoostingMatrix
from classes.dataset import Dataset
from classes.graph import GraphPB
from classes.pattern_boosting import PatternBoosting
from data.load_dataset import split_dataset_by_metal_centers
from settings import Settings

# TODO remove logging info
if Settings().plot_log_info is True:
    import logging
    import tracemalloc

    logger = logging.getLogger(__name__)


def predict_test_dataset_graph(args):
    graph, pattern_boosting_models_list = args
    prediction = 0
    # counter is to normalize the predictions
    counter = 0

    for model in pattern_boosting_models_list:
        try:
            if model.trained is True:
                index = model.test_dataset.get_graphs_list().index(
                    graph) if graph in model.test_dataset.get_graphs_list() else -1
                if index != -1:
                    prediction += model.test_dataset_final_predictions[index]
                    counter += 1
        except:
            pass
    return prediction, counter


def predict_train_dataset_graph(args):
    graph, pattern_boosting_models_list = args
    prediction = 0.0
    counter = 0

    for model in pattern_boosting_models_list:
        try:
            index = model.training_dataset.get_graphs_list().index(graph)
            prediction += model.training_dataset_final_predictions[index]
            counter += 1
        except:
            pass
    return prediction, counter


def train_pattern_boosting(input_from_parallelization: tuple):
    pattern_boosting_model: PatternBoosting = input_from_parallelization[0]
    train_dataset = input_from_parallelization[1]
    test_dataset = input_from_parallelization[2]
    global_labels_variance = input_from_parallelization[3]
    pattern_boosting_model.training(train_dataset, test_dataset,
                                    global_train_labels_variance=global_labels_variance)
    return pattern_boosting_model


class WrapperPatternBoosting:
    def __init__(self, pattern_boosting_list: list = None, metal_center_list: list = None,
                 settings: Settings = Settings()):
        if metal_center_list is None:
            raise ValueError('metal_center_list cannot be None')
        if pattern_boosting_list is None:
            pattern_boosting_list = [PatternBoosting(settings=settings) for _ in range(len(metal_center_list))]

        if len(pattern_boosting_list) != len(metal_center_list):
            raise ValueError("not enough models for each metal center")
        self.pattern_boosting_models_list: list[PatternBoosting] = pattern_boosting_list
        self.metal_center_list = metal_center_list
        self.test_dataset: Dataset | None = None
        self.train_dataset: Dataset | None = None
        self.settings = copy.deepcopy(settings)
        self.total_boosting_matrix = None
        self.trained = False


    def train(self, train_dataset, test_dataset=None):
        # ----------------------------------------------------------------------------------------------------------
        # TODO remove memory tracer
        # TODO remove memory tracer
        if self.settings.plot_log_info is True:
            traced_memory = [memory_value / 1000000 for memory_value in tracemalloc.get_traced_memory()]
            logger.debug(f"Memory at the beginning of the prediction: {traced_memory}")
        # ----------------------------------------------------------------------------------------------------------

        if self.trained is False:

            # some checks for the input format, whether the input dataset it is already divided by metal centers or not
            if not isinstance(train_dataset, list):
                if not isinstance(train_dataset, Dataset):
                    train_dataset = Dataset(train_dataset, settings=self.settings)
                train_datasets_list = split_dataset_by_metal_centers(dataset=train_dataset,
                                                                     considered_metal_centers=self.metal_center_list)
                if test_dataset is not None:
                    if not isinstance(test_dataset, Dataset):
                        test_dataset = Dataset(test_dataset, settings=self.settings)
                    test_datasets_list = split_dataset_by_metal_centers(dataset=test_dataset,
                                                                        considered_metal_centers=self.metal_center_list)
            else:
                train_datasets_list = train_dataset
                test_datasets_list = test_dataset
            self.trained = True

        if test_dataset is None:
            test_datasets_list = [None for _ in range(len(train_datasets_list))]

        self.test_dataset_list = test_datasets_list

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        global_labels = [dataset.get_labels() for dataset in train_datasets_list if dataset is not None]
        # flattern the global variables list
        global_labels = [item for sublist in global_labels for item in sublist]
        global_labels_variance = np.var(global_labels)
        global_labels_variance = np.repeat(global_labels_variance, len(train_datasets_list))

        # ----------------------------------------------------------------------------------------------------------
        # TODO remove memory tracer
        if self.settings.plot_log_info is True:
            traced_memory = [memory_value / 1000000 for memory_value in tracemalloc.get_traced_memory()]
            logger.debug(f"Memory at the beginning of the parallelization: {traced_memory}")
        # ----------------------------------------------------------------------------------------------------------

        # Parallelization
        # ------------------------------------------------------------------------------------------------------------

        input_for_parallelization = list(zip(self.pattern_boosting_models_list, train_datasets_list, test_datasets_list,
                                             global_labels_variance))
        print(f"{len(input_for_parallelization)=}")
        with mp.get_context("spawn").Pool(17) as pool:
            # mp.get_context("spawn").Pool(min(self.settings.max_number_of_cores, len(self.settings.considered_metal_centers)))
            # pool = mp.Pool(min(self.settings.max_number_of_cores, len(self.settings.considered_metal_centers)))
            array_of_outputs = pool.map(train_pattern_boosting, input_for_parallelization)

        # -------------------------------------------------------------------------------------------------------------

        self.pattern_boosting_models_list = array_of_outputs

        # ----------------------------------------------------------------------------------------------------------
        # TODO remove memory tracer
        if self.settings.plot_log_info is True:
            traced_memory = [memory_value / 1000000 for memory_value in tracemalloc.get_traced_memory()]
            logger.debug(f"Memory at the end of parallelization of the prediction: {traced_memory}")
        # ----------------------------------------------------------------------------------------------------------

        if self.settings.show_analysis is True or self.settings.save_analysis is True:
            if test_dataset is not None:
                self.test_error = self.get_wrapper_test_error()
            self.train_error = self.get_wrapper_train_error()

        return array_of_outputs

    def re_train(self):

        train_datasets_list = [None] * len(self.pattern_boosting_models_list)
        test_datasets_list = self.test_dataset_list
        global_labels_variance = [None] * len(self.pattern_boosting_models_list)
        # Parallelization
        # ------------------------------------------------------------------------------------------------------------

        input_for_parallelization = zip(self.pattern_boosting_models_list, train_datasets_list, test_datasets_list,
                                        global_labels_variance)
        pool = mp.Pool(min(self.settings.max_number_of_cores, len(self.settings.considered_metal_centers)))
        array_of_outputs = pool.map(
            functools.partial(train_pattern_boosting), input_for_parallelization)
        # -------------------------------------------------------------------------------------------------------------
        if self.settings.show_analysis is True or self.settings.save_analysis is True:
            self.test_error = self.get_wrapper_test_error()
            self.train_error = self.get_wrapper_train_error()

        return array_of_outputs



    def predict_test_dataset_parallel(self) -> List[float] | None:
        if self.test_dataset is None:
            warnings.warn("Test dataset not found")
            return None
        else:
            graphs_list = self.test_dataset.get_graphs_list()

            # Define the number of processes to use, e.g., the number of CPU cores
            num_processes = min(len(graphs_list), self.settings.max_number_of_cores)

            # Create arguments list for multiprocessing
            args_list = [(graph, self.get_trained_pattern_boosting_models()) for graph in graphs_list]

            # Use multiprocessing Pool to parallelize the task
            with mp.get_context("spawn").Pool(4) as pool:
                results = pool.map(predict_test_dataset_graph, args_list)

            # Aggregation of results and normalization
            predictions = [result[0] for result in results]
            counters = [result[1] for result in results]

            # Normalizing the predictions with the counters
            for i in range(len(predictions)):
                if counters[i] != 0:
                    predictions[i] /= counters[i]

            return predictions

    def predict_train_dataset_parallel(self) -> List[float] | None:
        if self.train_dataset is None:
            warnings.warn("Train dataset not found")
            return None
        else:
            graphs_list = self.train_dataset.get_graphs_list()

            # Define the number of processes to use, e.g., the number of CPU cores
            num_processes = min(len(graphs_list), self.settings.max_number_of_cores)

            # Create arguments list for multiprocessing
            args_list = [(graph, self.get_trained_pattern_boosting_models()) for graph in graphs_list]

            # Use multiprocessing Pool to parallelize the task
            with mp.Pool(num_processes) as pool:
                results = pool.map(predict_train_dataset_graph, args_list)

            # Aggregation of results and normalization
            predictions = [result[0] for result in results]
            counters = [result[1] for result in results]

            # Normalizing the predictions with the counters
            for i in range(len(predictions)):
                if counters[i] != 0:
                    predictions[i] /= counters[i]

            return predictions

    def predict_test_dataset(self) -> List[float] | None:
        if self.test_dataset is None:
            warnings.warn("Test dataset not found")
            return None
        else:
            predictions = [0.0] * len(self.test_dataset.get_graphs_list())
            counters = [0] * len(self.test_dataset.get_graphs_list())
            for i, graph in enumerate(self.test_dataset.get_graphs_list()):

                for model in self.pattern_boosting_models_list:
                    try:
                        index = model.test_dataset.get_graphs_list().index(graph)
                        predictions[i] += model.test_dataset_final_predictions[index]
                        counters[i] += 1

                    except:
                        pass
            for i in range(len(predictions)):
                if counters[i] != 0:
                    predictions[i] = predictions[i] / counters[i]
            return predictions

    def predict_train_dataset(self) -> List[float] | None:
        if self.train_dataset is None:
            warnings.warn("Train dataset not found")
            return None
        else:
            predictions = [0.0] * len(self.train_dataset.get_graphs_list())
            counters = [0] * len(self.train_dataset.get_graphs_list())
            for i, graph in enumerate(self.train_dataset.get_graphs_list()):
                for model in self.pattern_boosting_models_list:
                    try:
                        index = model.training_dataset.get_graphs_list().index(graph)
                        predictions[i] += model.training_dataset_final_predictions[index]
                        counters[i] += 1
                    except:
                        pass
            for i in range(len(predictions)):
                if counters[i] != 0:
                    predictions[i] = predictions[i] / counters[i]

            return predictions

    def predict(self, graphs_list, boosting_matrix_matrix=None):
        # If a graph has more metal centers ,the final prediction will be just the average between the different models
        if isinstance(graphs_list, GraphPB):
            graphs_list = [graphs_list]
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        prediction = [None] * len(graphs_list)
        for i, graph in enumerate(graphs_list):
            metal_centers_labels = [graph.node_to_label[metal_center] for metal_center in graph.metal_center]
            graph_prediction = 0
            counter = 0
            for metal_label in metal_centers_labels:
                try:
                    index = self.metal_center_list.index(metal_label)
                    if boosting_matrix_matrix is None:
                        graph_prediction = graph_prediction + self.pattern_boosting_models_list[index].predict(graph)
                    else:
                        # if we already have a boosting matrix
                        graph_prediction = graph_prediction + self.pattern_boosting_models_list[index].predict(graph, [
                            boosting_matrix_matrix[i]])
                    counter += 1
                except:
                    pass

            graph_prediction = graph_prediction / counter
            prediction[i] = graph_prediction
        # Concatenate the arrays into a single array
        concatenated_array = np.concatenate(prediction)

        # Convert the resulting array to a Python list
        prediction = concatenated_array.tolist()
        return prediction

    def get_wrapper_test_error(self) -> np.array:

        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_test_models_errors(), dataset="test")

    # need to fix the fact that the tested model errors if not trained they return [], not a list of errors, so everything here does not work
    def get_wrapper_train_error(self) -> np.array:
        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_train_models_errors(), dataset="train")

    @staticmethod
    def __weighted_average(error_lists: List[List[float]], weights: List[int]) -> List[Optional[float]]:
        """
        Calculate the weighted average of the i-th elements in a list of error_lists, given a list of weights.
        The function handles error_lists of different lengths, excluding error_lists that don't have an i-th element.

        :param error_lists: A list of lists, where each error_list contains float elements.
        :param weights: A list of integers representing the importance (weight) of each error_list.
                        The weights do not need to sum up to 1.
        :return: A list of floats, where the i-th element is the weighted average of the i-th elements from the error_lists.
                 If no error_lists contain an i-th element, the i-th position in the output list will be None.

        Example:
            error_lists = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0, 7.0, 8.0, 9.0]]
            weights = [1, 2, 3]
            output = weighted_average_of_error_lists(error_lists, weights)
            print(output)  # Output: [4.5, 5.333333333333333, 6.75, 9.0]
        """
        weighted_avgs = []
        max_len = max(len(single_model_error) for single_model_error in error_lists)

        for i in range(max_len):
            weighted_sum = 0.0
            total_weight = 0

            for single_model_error, weight in zip(error_lists, weights):
                if i < len(single_model_error):
                    weighted_sum += single_model_error[i] * weight
                    total_weight += weight

            if total_weight != 0:
                weighted_avg = weighted_sum / total_weight
            else:
                weighted_avg = None

            weighted_avgs.append(weighted_avg)

        return weighted_avgs

    def __get_average_of_matrix_of_nested_list_of_errors(self, errors_lists: Iterable[Sequence[float]], dataset: str,
                                                         mode='average') -> np.array:
        '''
        :param errors_lists:
        :param mode: a string 'average', 'max', 'min' to indicate which error measure it should be considered. Average, max or min of the all error made at the i-th step
        :return: an array where each cell correspond to one iteration, since we have multiple base learners each
                    iteration is actually n_base learners iterations, so the final array returned contains the error repeated
                    n_iterations times
        '''

        weights = self.get_number_of_observations_per_model(dataset_name=dataset)

        # filter out the models who are not trained (because their metal center is not contained in the training dataset)

        errors_lists, weights = list(zip(*[(error, weights) for error, weights in zip(errors_lists, weights) if
                                           not (error is None or error == [])]))

        if mode == 'average':
            error = np.array(self.__weighted_average(errors_lists, weights))
            number_of_trained_models = len(self.pattern_boosting_models_list)
        else:
            raise TypeError('Uknown Mode')

        # TODO return the error based on the parameter 'mode' that can be average, max, min
        return error

    def get_number_of_trained_models(self):
        return len(self.pattern_boosting_models_list)

    def get_train_models_errors(self) -> Iterable[Sequence[float]]:
        '''
        :return: a nested list where each row is the vector of errors coming from the model
        '''

        train_model_errors = [model.train_error for model in self.get_trained_pattern_boosting_models()]
        return train_model_errors

    def get_test_models_errors(self) -> Iterable[Sequence[float]]:
        '''

        :return: a nested list where each row is the vector of errors coming from the model
        '''
        test_model_errors = [model.test_error for model in self.get_trained_pattern_boosting_models()]
        return test_model_errors

    def get_number_of_observations_per_model(self, dataset_name: str) -> list[int]:
        '''
        :param dataset_name: "training" or "test" depending on which of the two we want the observations to come from
        :return: the number of observations used for the training/test of each model
        '''
        if dataset_name == "train" or dataset_name == "test" or dataset_name == "training" or dataset_name == "testing":
            dimension_list = [model.get_dataset(dataset_name).get_dimension() for model in
                              self.pattern_boosting_models_list]
            return dimension_list
        else:
            raise TypeError(
                f"tipe of dataset must be 'train' or 'test' or 'training' or 'testing', got {dataset_name} instead")

    def evaluate(self, dataset: Dataset, boosting_matrix_matrix=None):

        y_pred = self.predict(dataset.get_graphs_list(), boosting_matrix_matrix)
        labels = dataset.get_labels()

        if self.settings.final_evaluation_error == "MSE":
            model_error = metrics.mean_squared_error(labels, y_pred)
        elif self.settings.final_evaluation_error == "absolute_mean_error":
            model_error = metrics.mean_absolute_error(labels, y_pred)
        else:
            raise ValueError("measure error not found")
        return model_error

    def get_pattern_boosting_models(self):
        return self.pattern_boosting_models_list

    def get_trained_pattern_boosting_models(self):
        return [model for model in self.get_pattern_boosting_models() if model.trained is True]

    def create_boosting_matrices_for(self, graphs_list, convert_to_boosting_matrix=False, selected_paths=None) -> list[
                                                                                                                      np.array] | \
                                                                                                                  list[
                                                                                                                      BoostingMatrix]:
        '''
        :param graphs_list: list or dataset of graphs
        :param convert_to_boosting_matrix: to decide if at the end the boosting matrices should be converted in the class BoostingMatrix
        :return: a list of matrices, every matrix is the boosting matrix corresponding to one metal center's model
                 (N.B. the order is preserved so the matrix can be used by the model for predictions)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        matrices_list = [
            model.create_boosting_matrix_for(graphs_list, convert_to_boosting_matrix, selected_paths=selected_paths) for
            model in
            self.pattern_boosting_models_list if model.trained is True]
        return matrices_list

    def create_ordered_boosting_matrix(self, graphs_list: list | Dataset,
                                       convert_to_boosting_matrix: bool = False,
                                       selected_paths=None) -> np.ndarray | BoostingMatrix:
        '''
        :param graphs_list: list or dataset of graphs
        :param convert_to_boosting_matrix: to decide if at the end the boosting matrices should be converted in the class BoostingMatrix
        :return: It returns one boosting matrix, with columns ordered by the relative pattern importance
                 (!!N.B. ti matrix can not be used for predictions since the columns are permuted respect to the original ordering!!)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()

        boosting_matrices_list = self.create_boosting_matrices_for(graphs_list, selected_paths=selected_paths)
        boosting_matrix = np.hstack(boosting_matrices_list)

        list_columns_importance = [model.get_boosting_matrix_columns_importance_values for model in
                                   self.pattern_boosting_models_list if model.trained is True]

        columns_importance = np.hstack(list_columns_importance)

        # here we order the boosting matrix by the importance value of the columns
        transposed_boosting_matrix = boosting_matrix.transpose()
        boosting_matrix = [column for _, column in sorted(zip(columns_importance, transposed_boosting_matrix))]
        boosting_matrix = np.array(boosting_matrix)
        boosting_matrix + boosting_matrix.transpose()
        if convert_to_boosting_matrix is False:
            return boosting_matrix
        else:
            # TODO test this part
            headers = [model.get_boosting_matrix_header() for model in self.pattern_boosting_models_list if
                       model.trained is True]
            headers = self.__flatten_concatenation(headers)
            ordered_headers = [header for _, _, header in sorted(zip(headers, transposed_boosting_matrix, headers))]
            return BoostingMatrix(boosting_matrix, ordered_headers,
                                  sorted(columns_importance))

    def get_train_correlations(self, paths_list: list) -> dict:
        correlations_dictionaries = [model.get_max_path_correlation(paths_list) for model in
                                     self.pattern_boosting_models_list if model.trained is True]
        result = defaultdict(int)
        for dictionary in correlations_dictionaries:
            for key, value in dictionary.items():
                result[key] = max(result[key], value)
        return dict(result)

    @staticmethod
    def __flatten_concatenation(matrix):
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    def get_normalized_patterns_importance(self) -> Tuple[List[Tuple[int]], List[float]]:
        paths, importances = self.get_patterns_importance()
        norm_importances = [100 * float(i) / max(importances) for i in importances]
        return paths, norm_importances

    def get_patterns_importance(self) -> Tuple[List[Tuple[int]], List[float]]:
        '''
        It returns a list wih the patterns and a list with their importance
        '''

        importance = []
        paths = []

        for model in self.get_trained_pattern_boosting_models():
            paths_importance = np.array(model.get_boosting_matrix_columns_importance_values())
            paths_importance = np.array(
                paths_importance * model.get_dataset_dimension('training')) / self.train_dataset.get_dimension()

            importance += list(paths_importance)
            paths += model.get_boosting_matrix_header()
        return paths, importance

    def get_selected_paths(self) -> list:
        # note it returns all the paths that have been actually used by at least one model
        paths_length = sum(
            len(model.get_selected_paths_in_boosting_matrix()) for model in self.get_trained_pattern_boosting_models())

        paths = [()] * paths_length

        paths_index = 0

        for model in self.get_trained_pattern_boosting_models():
            model_paths = model.get_selected_paths_in_boosting_matrix()

            paths[paths_index:paths_index + len(model_paths)] = model_paths

            paths_index += len(model_paths)
        return list(set(paths))

    def get_boosting_matrix_header(self) -> list:
        # it returns the list of selected paths
        paths_in_header = set()
        for model in self.get_trained_pattern_boosting_models():
            model_paths = model.get_selected_paths_in_boosting_matrix()
            paths_in_header.update(model_paths)

        return list(paths_in_header)

    def get_test_error_per_number_of_base_learners(self):

        n_trained_models = len(self.get_trained_pattern_boosting_models())

        output_list = []

        if n_trained_models <= 1:
            # If n is 1 or less no interpolation is required.
            return self.test_error

        for i in range(len(self.test_error) - 1):
            current_value = self.test_error[i]
            next_value = self.test_error[i + 1]
            step = (next_value - current_value) / n_trained_models
            for j in range(n_trained_models):  # Note: this will only add n-1 interpolated values
                output_list.append(current_value + j * step)

        # Ensure that the last element of the input list is repeated n times
        last_value = self.test_error[-1]
        for i in range(n_trained_models):
            output_list.append(last_value)

        return output_list

    def get_train_error_per_number_of_base_learners(self):

        n_trained_models = len(self.get_trained_pattern_boosting_models())

        output_list = []

        if n_trained_models <= 1:
            # If n is 1 or less no interpolation is required.
            return self.train_error

        for i in range(len(self.train_error) - 1):
            current_value = self.train_error[i]
            next_value = self.train_error[i + 1]
            step = (next_value - current_value) / n_trained_models
            for j in range(n_trained_models):  # Note: this will only add n-1 interpolated values
                output_list.append(current_value + j * step)

        # Ensure that the last element of the input list is repeated n times
        last_value = self.train_error[-1]
        for i in range(n_trained_models):
            output_list.append(last_value)

        return output_list

    def merge_models(self, wrapper_boosting_models_list: list[Self]):
        # we assume this is a trained wrapped boosting model but with no real training on the pattern boosting models. so we take the pattern boosting models from others wrapper pattern boosting, the challange is to insert the pattern boosting models in the right order.
        for wrapper_model in wrapper_boosting_models_list:
            for i, pattern_boosting_model in enumerate(wrapper_model.get_pattern_boosting_models()):
                if pattern_boosting_model.trained is True:
                    self.pattern_boosting_models_list[i] = pattern_boosting_model
