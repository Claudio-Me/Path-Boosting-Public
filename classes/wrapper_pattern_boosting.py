# Takes as input a lyst of parrent boosting models and "merge" them into one
from settings import Settings
from classes.dataset import Dataset
from classes.graph import GraphPB
from data.load_dataset import split_dataset_by_metal_centers
from classes.pattern_boosting import PatternBoosting
from multiprocessing.dummy import Pool as ThreadPool
import functools
import numpy as np
from sklearn import metrics
from typing import Sequence, Iterable


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

        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_test_models_errors())

    # need to fix the fact that the tested model errors if not trained they return [], not a list of errors, so everithing here does not work
    def get_wrapper_train_error(self) -> Iterable[float]:
        return self.__get_average_of_matrix_of_nested_list_of_errors(self.get_train_models_errors())

    @staticmethod
    def __get_average_of_matrix_of_nested_list_of_errors(errors_lists) -> np.array:
        '''
        :param errors_lists:
        :return: an array where each cell correspond to one iteration, since we have multiple base learners each
                    iteration is actually n_base learners iterations, so the final array returned contains the error repeated
                    n_iterations times
        '''

        # filter out the models who are not trained (because their metal center is not contained in the training dataset)
        errors_lists = [error for error in errors_lists if not (error is None or error == [])]
        number_of_trained_models = len(errors_lists)
        # FIXME check that the lengths of the error for each model is the same (for now it always is
        #  because we set Setting.train_target_error to zero, but if not, then the train for some
        #  models may stop earlyer)

        errors_lists = np.array(errors_lists)

        # array of errors:
        error = np.mean(errors_lists, axis=0)

        # here we just repeat the errors by the number of trained models
        error = np.repeat(error, number_of_trained_models)

        return error

    def get_train_models_errors(self) -> Iterable[Sequence[float]]:
        # it returns a nested list where each row is the vector of errors coming from the model
        train_model_errors = [model.train_error for model in self.pattern_boosting_models_list]
        return train_model_errors

    def get_test_models_errors(self) -> Iterable[Sequence[float]]:
        # it returns a nested list where each row is the vector of errors coming from the model
        test_model_errors = [model.test_error for model in self.pattern_boosting_models_list]
        return test_model_errors

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

    def create_boosting_matrices_for(self, graphs_list):
        '''
        :param graphs_list: list or dataset of graphs
        :return: a list of matrices, every matrix is the boosting matrix corresponding to one metal center's model
                 (N.B. the order is preserved so the matrix can be used by the model for predictions)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        matrices_list = [model.create_boosting_matrix_for(graphs_list) for model in self.pattern_boosting_models_list]
        return matrices_list

    def create_ordered_booting_matrix(self, graphs_list):
        '''
        :param graph_list: list or dataset of graphs
        :return: It returns one boosting matrix, with columns ordered by the relative pattern importance
                 (!!N.B. ti matrix can not be used for predictions since the columns are permuted respect to the original ordering!!)
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()

        boosting_matrices_list = self.create_boosting_matrices_for(graphs_list)
        list_columns_importance = [model.get_boosting_matrix_columns_importance_values for model in
                                   self.pattern_boosting_models_list]
