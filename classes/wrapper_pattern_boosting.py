# Takes as input a lyst of parrent boosting models and "merge" them into one
from settings import Settings
from classes.dataset import Dataset
from classes.graph import GraphPB
from classes.boosting_matrix import BoostingMatrix
from data.load_dataset import split_dataset_by_metal_centers
from classes.pattern_boosting import PatternBoosting
from multiprocessing.dummy import Pool as ThreadPool
import functools
import numpy as np
from sklearn import metrics
from typing import Sequence, Iterable
from typing import List, Tuple


class WrapperPatternBoosting:
    def __init__(self, pattern_boosting_list: list = None, metal_center_list: list = Settings.considered_metal_centers):
        if pattern_boosting_list is None:
            pattern_boosting_list = [PatternBoosting() for _ in range(len(metal_center_list))]

        if len(pattern_boosting_list) != len(metal_center_list):
            raise ValueError("not enough models for each metal center")
        self.pattern_boosting_models_list: list[PatternBoosting] = pattern_boosting_list
        self.metal_center_list = metal_center_list

    def predict(self, graphs_list, boosting_matrix_matrix=None):
        # If a graph has more metal centers ,the final prediction will be just the average between the different models
        if isinstance(graphs_list, GraphPB):
            graphs_list = [graphs_list]
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        prediction = [] * len(graphs_list)
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
        return prediction

    @staticmethod
    def __train_pattern_boosting(input_from_parallelization: tuple):
        pattern_boosting_model: PatternBoosting = input_from_parallelization[0]
        train_dataset = input_from_parallelization[1]
        test_dataset = input_from_parallelization[2]
        pattern_boosting_model.training(train_dataset, test_dataset)
        return pattern_boosting_model

    def get_wrapper_test_error(self) -> Iterable[float]:

        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_test_models_errors(), dataset="test")

    # need to fix the fact that the tested model errors if not trained they return [], not a list of errors, so everything here does not work
    def get_wrapper_train_error(self) -> Iterable[float]:
        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_train_models_errors(), dataset="train")

    @staticmethod
    def __weighted_average(errors_lists, weights):
        '''
        Parameters:
        errors_lists (list of lists): A nested list where each sublist contains numerical values.
                    The number of elements in each sublist should be equal.
        weights (list): A list of numerical values serving as weights for the errors_lists.
                    The number of weights should be equal to the number of errors_lists.

        Returns:
        list: A list containing the weighted averages of the i-th elements of each sublist.
              If the lengths of errors_lists and weights are not equal, an error message is returned.

        Raises:
        TypeError: If errors_lists is not a list or weights is not a list.
        ValueError: If all elements in each sublist are not numeric.
        ValueError: If the weights do not sum up to 1.

        Example:
         errors_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
         weights = [0.3, 0.2, 0.5]
         weighted_average(errors_lists, weights)
        [5.0, 6.0, 7.0]
        '''
        if len(errors_lists) == len(weights):
            return [sum(x * y for x, y in zip(sublist, weights)) / sum(weights) for sublist in zip(*errors_lists)]
        else:
            raise TypeError("The length of errors_lists and weights should be equal")

    def __get_average_of_matrix_of_nested_list_of_errors(self, errors_lists: Iterable[Sequence[float]], dataset: str,
                                                         mode='average') -> np.array:
        '''
        :param errors_lists:
        :param mode: a string 'average', 'max', 'min' to indicate which error measure it should be considered. Average, max or min of the all error made at the i-th step
        :return: an array where each cell correspond to one iteration, since we have multiple base learners each
                    iteration is actually n_base learners iterations, so the final array returned contains the error repeated
                    n_iterations times
        '''

        weights = self.get_number_of_observations_per_model(dataset=dataset)

        # filter out the models who are not trained (because their metal center is not contained in the training dataset)

        errors_lists, weights = list(zip(*[(error, weights) for error, weights in zip(errors_lists, weights) if
                                           not (error is None or error == [])]))

        error = np.array(self.__weighted_average(errors_lists, weights))
        number_of_trained_models = len(self.pattern_boosting_models_list)


        # TODO return the error based on the parameter 'mode' that can be average, max, min
        return error

    def get_train_models_errors(self) -> Iterable[Sequence[float]]:
        '''
        :return: a nested list where each row is the vector of errors coming from the model
        '''

        train_model_errors = [model.train_error for model in self.pattern_boosting_models_list]
        return train_model_errors

    def get_test_models_errors(self) -> Iterable[Sequence[float]]:
        '''

        :return: a nested list where each row is the vector of errors coming from the model
        '''
        test_model_errors = [model.test_error for model in self.pattern_boosting_models_list]
        return test_model_errors

    def get_number_of_observations_per_model(self, dataset: str) -> list[int]:
        '''
        :param dataset: "training" or "test" depending on which of the two we want the observations to come from
        :return: the number of observations used for the training/test of each model
        '''
        if dataset == "train" or dataset == "test" or dataset == "training" or dataset == "testing":
            dimension_list = [model.get_dataset(dataset).get_dimension() for model in self.pattern_boosting_models_list]
            return dimension_list
        else:
            raise TypeError(
                f"tipe of dataset must be 'train' or 'test' or 'training' or 'testing', got {dataset} instead")

    def train(self, train_dataset, test_dataset=None):

        # some checks for the input format, whether the input dataset it is already divided by metal centers or not
        if not isinstance(train_dataset, list):
            train_datasets_list = split_dataset_by_metal_centers(dataset=train_dataset,
                                                                 considered_metal_centers=self.metal_center_list)
            if test_dataset is not None:
                test_datasets_list = split_dataset_by_metal_centers(dataset=test_dataset,
                                                                    considered_metal_centers=self.metal_center_list)
        else:
            train_datasets_list = train_dataset
            test_datasets_list = test_dataset

        if test_dataset is None:
            test_datasets_list = [None for _ in range(len(train_datasets_list))]

        # Parallelization
        # ------------------------------------------------------------------------------------------------------------

        input_for_parallelization = zip(self.pattern_boosting_models_list, train_datasets_list, test_datasets_list)
        pool = ThreadPool(min(10, len(Settings.considered_metal_centers)))
        array_of_outputs = pool.map(
            functools.partial(self.__train_pattern_boosting), input_for_parallelization)
        # -------------------------------------------------------------------------------------------------------------
        self.test_error = self.get_wrapper_test_error()
        self.train_error = self.get_wrapper_train_error()

        return array_of_outputs

    def evaluate(self, dataset: Dataset, boosting_matrix_matrix=None):

        y_pred = self.predict(dataset.get_graphs_list(), boosting_matrix_matrix)
        labels = dataset.get_labels()

        if Settings.final_evaluation_error == "MSE":
            model_error = metrics.mean_squared_error(labels, y_pred)
        elif Settings.final_evaluation_error == "absolute_mean_error":
            model_error = metrics.mean_absolute_error(labels, y_pred)
        else:
            raise ValueError("measure error not found")
        return model_error

    def get_pattern_boosting_models(self):
        return self.pattern_boosting_models_list

    def get_trained_pattern_boosting_models(self):
        return [model for model in self.get_pattern_boosting_models() if model.trained is True]

    def create_boosting_matrices_for(self, graphs_list, convert_to_boosting_matrix=False) -> list[
                                                                                                 np.array] | list[
                                                                                                 BoostingMatrix]:
        '''
        :param graphs_list: list or dataset of graphs
        :param convert_to_boosting_matrix: to decide if at the end the boosting matrices should be converted in the class BoostingMatrix
        :return: a list of matrices, every matrix is the boosting matrix corresponding to one metal center's model
                 (N.B. the order is preserved so the matrix can be used by the model for predictions)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        matrices_list = [model.create_boosting_matrix_for(graphs_list, convert_to_boosting_matrix) for model in
                         self.pattern_boosting_models_list]
        return matrices_list

    def create_ordered_boosting_matrix(self, graphs_list,
                                       convert_to_boosting_matrix=False) -> np.ndarray | BoostingMatrix:
        '''
        :param graphs_list: list or dataset of graphs
        :param convert_to_boosting_matrix: to decide if at the end the boosting matrices should be converted in the class BoostingMatrix
        :return: It returns one boosting matrix, with columns ordered by the relative pattern importance
                 (!!N.B. ti matrix can not be used for predictions since the columns are permuted respect to the original ordering!!)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()

        boosting_matrices_list = self.create_boosting_matrices_for(graphs_list)
        boosting_matrix = np.hstack(boosting_matrices_list)

        list_columns_importance = [model.get_boosting_matrix_columns_importance_values for model in
                                   self.pattern_boosting_models_list]

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
            headers = [model.get_boosting_matrix_header() for model in self.pattern_boosting_models_list]
            headers = self.__flatten_concatenation(headers)
            ordered_headers = [header for _, _, header in sorted(zip(headers, transposed_boosting_matrix, headers))]
            return BoostingMatrix(boosting_matrix, ordered_headers,
                                  sorted(columns_importance))

    @staticmethod
    def __flatten_concatenation(matrix):
        flat_list = []
        for row in matrix:
            flat_list += row
        return flat_list

    def get_patterns_importance(self) -> Tuple[List[Tuple[int]], List[float]]:
        '''
        It returns a list wih the patterns and a list with their importance
        # extremely optimized version of this:
        importance = []
        paths = []

        for model in self.get_pattern_boosting_models():
            importance += model.get_boosting_matrix_columns_importance_values()
            paths += model.get_boosting_matrix_header()
        '''

        importance_length = sum(
            len(model.get_boosting_matrix_columns_importance_values()) for model in self.get_trained_pattern_boosting_models())
        paths_length = sum(len(model.get_boosting_matrix_header()) for model in self.get_trained_pattern_boosting_models())

        importance = [0.0] * importance_length
        paths = [()] * paths_length

        importance_index = 0
        paths_index = 0

        for model in self.get_trained_pattern_boosting_models():
            model_importance = model.get_boosting_matrix_columns_importance_values()
            model_paths = model.get_boosting_matrix_header()

            importance[importance_index:importance_index + len(model_importance)] = model_importance
            paths[paths_index:paths_index + len(model_paths)] = model_paths

            importance_index += len(model_importance)
            paths_index += len(model_paths)
        return paths, importance
