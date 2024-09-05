import warnings
from classes.graph import GraphPB
from classes.boosting_matrix import BoostingMatrix
from settings import Settings
from classes.gradient_boosting_step import GradientBoostingStep
from classes.dataset import Dataset
from collections import defaultdict
from classes.enumeration.estimation_type import EstimationType
from classes.gradient_boosting_model import GradientBoostingModel
import numpy as np
import multiprocessing
import functools
import itertools
import sys
import copy
from typing import List, Tuple
import numpy.typing as npt

# from pympler import asizeof

if Settings.parallelization is True:
    from mpi4py import MPI


class PatternBoosting:
    def __init__(self, settings=Settings(), model: GradientBoostingModel = None):
        self.settings = copy.deepcopy(settings)
        self.settings.target_paths = copy.deepcopy(self.settings.target_paths)
        self.model = model
        self.trained = False
        self.test_error: list[float] = []
        self.train_error: list[float] = []
        self.average_path_length = []
        self.number_of_learners = []
        self.gradient_boosting_step = GradientBoostingStep()
        self.n_iterations = None
        self.boosting_matrix_matrix_for_test_dataset = None
        self.global_train_labels_variance = None

    def training(self, training_dataset, test_dataset=None, global_train_labels_variance=None):
        """Trains the model, it is possible to call this function multiple times, in this case the dataset used for
        training is always the one took as input the first time the function "training" is called
        In future versions it will be possible to give as input a new dataset"""

        if isinstance(training_dataset, Dataset):
            self.training_dataset = training_dataset

        elif isinstance(training_dataset, list):
            self.training_dataset = Dataset(training_dataset)

        elif training_dataset is None:
            if self.trained is False:
                return
        else:
            raise TypeError("Input dataset not recognized")

        if test_dataset is not None:
            if isinstance(test_dataset, Dataset):
                self.test_dataset = test_dataset

            elif isinstance(test_dataset, list):
                self.test_dataset = Dataset(test_dataset)
            else:
                raise TypeError("Input test dataset not recognized")
        else:
            self.test_dataset = None

        # if it is the first time we train this model
        if self.trained is False:
            self.trained = True
            self.__initialize_boosting_matrix()
            self.global_train_labels_variance = global_train_labels_variance

        else:
            if training_dataset is not None:
                boosting_matrix_matrix = np.array([self.__create_boosting_vector_for_graph(graph) for graph in
                                                   training_dataset.graphs_list])
                self.boosting_matrix = BoostingMatrix(boosting_matrix_matrix, self.boosting_matrix.header,
                                                      self.boosting_matrix.columns_importance)

        # cumulative number of paths that has been selected at each step
        self.n_selected_paths = []
        if self.settings.dataset_name == "5k_synthetic_dataset":
            target_paths = copy.deepcopy(self.settings.target_paths)

            # (numero totale di "target paths" trovati fino ad ora) / (numero totale di path selezionati senza contare le ripetizioni)
            self.true_positive_ratio_1 = []
            times_a_target_path_has_been_selected = 0

            already_spotted_paths = set()
            n_spotted_paths = 0

        for iteration_number in range(self.settings.maximum_number_of_steps):
            if self.settings.verbose is True:
                print("Step number ", iteration_number + 1)
            # print("size of pattern boosting: ", asizeof.asizeof(self))
            # print("size of trainig dataset: ", asizeof.asizeof(self.training_dataset))

            self.n_iterations = iteration_number + 1

            selected_column_number, self.model = self.gradient_boosting_step.select_column(model=self.model,
                                                                                           boosting_matrix=self.boosting_matrix,
                                                                                           labels=self.training_dataset.labels,
                                                                                           number_of_learners=iteration_number + 1)
            if test_dataset is not None and self.settings.algorithm != "Xgb_step":
                self.test_error.append(self.evaluate(test_dataset))

            # TODO improve this evaluation process to get train error, it seems unnecessary to re-evaluate the whole dataset
            if self.settings.algorithm != "Xgb_step":
                self.train_error.append(self.evaluate(self.training_dataset, self.boosting_matrix.matrix))
            else:
                self.train_error.append(self.model.get_last_training_error() * self.model.get_last_training_error())

            if self.settings.update_features_importance_by_comparison is False:
                if len(self.train_error) <= 1:
                    default_importance_value = np.var(training_dataset.labels)
                else:
                    default_importance_value = None

                self.boosting_matrix.update_pattern_importance_of_column_based_on_error_improvment(
                    selected_column_number,
                    train_error=self.train_error,
                    default_value=default_importance_value)
            else:
                if len(self.get_boosting_matrix_header()) <= 1:
                    # improvment = np.var(self.training_dataset.get_labels())
                    improvment = 0
                else:
                    second_best_column, second_column_error = self.find_second_best_column(selected_column_number)

                    improvment = second_column_error - self.train_error[-1]

                self.boosting_matrix.update_column_importance(selected_column_number, improvment)

            self.number_of_learners.append(iteration_number + 1)

            # expand boosting matrix
            self.__expand_boosting_matrix(selected_column_number)

            self.average_path_length.append(self.boosting_matrix.average_path_length())

            self.n_selected_paths.append(len(self.get_selected_paths_in_boosting_matrix()))

            if self.settings.dataset_name == "5k_synthetic_dataset":
                selected_path = self.boosting_matrix.get_path_associated_to_column(selected_column_number)

                if selected_path in target_paths:
                    times_a_target_path_has_been_selected += 1
                    if not (selected_path in already_spotted_paths):
                        n_spotted_paths += 1
                        already_spotted_paths.add(selected_path)
                    target_paths.remove(selected_path)

                n_selected_paths = len(self.get_selected_paths_in_boosting_matrix())
                self.true_positive_ratio_1.append(times_a_target_path_has_been_selected / n_selected_paths)

            if self.train_error[-1] < Settings.target_train_error:
                break

        self.training_dataset_final_predictions = self.predict(self.training_dataset, self.boosting_matrix)

        if test_dataset is not None and self.settings.algorithm == "Xgb_step":
            if self.settings.verbose is True:
                print("computing test boosting matrix")
                if self.settings.wrapper_boosting is True:
                    print("metal center ", self.training_dataset.graphs_list[0].get_metal_center_labels())
            self.boosting_matrix_matrix_for_test_dataset = self.create_boosting_matrix_for(
                test_dataset)
            if self.settings.verbose is True:
                print("predicting test dataset final error")
            self.test_dataset_final_predictions = self.predict(self.test_dataset,
                                                               self.boosting_matrix_matrix_for_test_dataset)
            if self.settings.verbose is True:
                print("evaluate progression")
            self.test_error = self.evaluate_progression(test_dataset, self.boosting_matrix_matrix_for_test_dataset)

    def find_second_best_column(self, first_column_number: int) -> tuple[int, float]:
        """
        Finds the second-best column to be used in the boosting model based on the feature importances.

        This method wraps around the model's `select_second_best_column` function. It provides an interface to retrieve the second most important
        feature after excluding a specified column. If the model's algorithm setting does not match "Xgb_step",
        a TypeError is raised.

        :param first_column_number: The index of the column to be excluded from feature importance assessment.
        :type first_column_number: int
        :raises TypeError: If the wrong algorithm is specified in the settings and the function is incompatible
                           with the model in use.
        :return: A tuple containing the index of the second-best column (with the second-highest feature
                 importance) and the final training error of the model if the algorithm setting is "Xgb_step".
        :rtype: tuple[int, float]
        """
        model = self.model
        boosting_matrix = self.boosting_matrix
        labels = self.training_dataset.labels

        if (
                self.settings.algorithm == "Xgb_step" or self.settings.algorithm == "decision_tree") and self.trained is True:
            return model.select_second_best_column(boosting_matrix, first_column_number, labels,
                                                   global_labels_variance=self.global_train_labels_variance)
        else:
            raise TypeError(
                "Wrong algorithm, impossible to get feature importance by comparison for this model; "
                "change update_features_importance_by_comparison in settings."
            )

    def evaluate_progression(self, dataset: Dataset, boosting_matrix_matrix=None):
        '''
        :param dataset: dataset to evaluate the progression of the model on.
        :param boosting_matrix_matrix: optional, boosting matrix of the dataset.
        :return: it returns an array  of the test error of the model in which the i-th corresponds to the performance of the model using only the first 'i' base learners.
        '''
        if self.trained is False:
            warnings.warn("This model is not trained")
            return None
        if boosting_matrix_matrix is None:
            boosting_matrix_matrix = self.create_boosting_matrix_for(dataset)
        test_error = self.model.evaluate_progression(boosting_matrix_matrix, dataset.labels)
        return test_error

    def predict(self, graphs_list, boosting_matrix_matrix=None):
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()
        if self.trained is False:
            warnings.warn("This model is not trained")
            return None
        if isinstance(graphs_list, GraphPB):
            graphs_list = [graphs_list]
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()

        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()
        if boosting_matrix_matrix is None:
            boosting_matrix_matrix = self.create_boosting_matrix_for(graphs_list)
        prediction = self.model.predict_my(boosting_matrix_matrix)
        return prediction

    def create_boosting_matrix_for(self, graphs_list, convert_to_boosting_matrix=False) -> np.ndarray | BoostingMatrix:
        '''
        :param convert_to_boosting_matrix: to decide if at the end the boosting matrices should be converted in the class BoostingMatrix
        :param graphs_list: list of graphs in dataset format or list
        :return: the boosting matrix of relative to the set of graphs given in input, the order of the columns is the
                 same as the order of the header of the model's boosting matrix
        '''
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        boosting_matrix_matrix_rows = []
        for graph in graphs_list:
            boosting_matrix_matrix_rows.append(self.__create_boosting_vector_for_graph(graph))

        # old way
        # boosting_matrix_matrix = np.array([self.__create_boosting_vector_for_graph(graph) for graph in graphs_list])
        boosting_matrix_matrix = np.array(boosting_matrix_matrix_rows)
        if convert_to_boosting_matrix is False:
            return boosting_matrix_matrix
        else:
            return BoostingMatrix(boosting_matrix_matrix, self.get_boosting_matrix_header(),
                                  self.get_boosting_matrix_columns_importance_values())

    def evaluate(self, dataset: Dataset, boosting_matrix_matrix=None):
        if self.trained is False:
            warnings.warn("This model is not trained")
            return None
        if boosting_matrix_matrix is None:
            boosting_matrix_matrix = self.generate_boosting_matrix(dataset)

        error = self.model.evaluate(boosting_matrix_matrix, dataset.labels)
        return error

    def generate_boosting_matrix(self, dataset: Dataset) -> np.array:
        boosting_matrix_matrix = np.array(
            [self.__create_boosting_vector_for_graph(graph) for graph in dataset.get_graphs_list()])
        return boosting_matrix_matrix

    def __create_boosting_vector_for_graph(self, graph: GraphPB) -> np.array:
        boosting_vector = [0] * len(self.boosting_matrix.get_header())
        for i, label_path in enumerate(self.boosting_matrix.get_header()):
            boosting_vector[i] = graph.number_of_time_path_is_present_in_graph(label_path)

        # old way
        # boosting_vector = [graph.number_of_time_path_is_present_in_graph(label_path) for label_path in self.boosting_matrix.get_header()]
        return np.array(boosting_vector)

    def __get_new_columns(self, new_paths, graphs_that_contain_selected_column_path):
        """
        given a set of paths and a set of graphs that contains the given paths it returns the new column that should
        be added to the dataset. in each line of the column the value represent the number of times the path that
        correspond to the column is present in the graph.
        The order of the columns follows the order of the input vector of paths
        """

        new_columns = np.zeros((len(self.training_dataset.graphs_list), len(new_paths)))

        """
        # old code, it works fine, but slow
        
        for path_number, path in enumerate(new_paths):
            for graph_number in graphs_that_contain_selected_column_path:
                graph = self.training_dataset.graphs_list[graph_number]
                n = graph.number_of_time_path_is_present_in_graph(path)
                new_columns[graph_number][path_number] = n
        """
        print("number of graphs that need to be considered ", len(graphs_that_contain_selected_column_path))
        print("number of paths that need to be considered ", len(new_paths))

        for path_number, path in enumerate(new_paths):
            print('path number ', path_number)
            column = np.array(
                [self.training_dataset.graphs_list[graph_number].number_of_time_path_is_present_in_graph(path) for
                 graph_number in graphs_that_contain_selected_column_path])

            for i, _ in enumerate(column):
                new_columns[graphs_that_contain_selected_column_path[i]][path_number] = column[i]

        return new_columns

    def __get_new_paths_and_columns(self, selected_path_label, graphs_that_contain_selected_column_path):
        new_paths = [list(
            self.training_dataset.graphs_list[graph_number].get_new_paths_labels_and_count(
                selected_path_label))
            for graph_number in graphs_that_contain_selected_column_path]

        counts = [graph_counts for graph_paths, graph_counts in new_paths]
        paths = [graph_paths for graph_paths, graph_counts in new_paths]
        new_paths = list(set([path for graph_paths in paths for path in graph_paths]))

        new_columns = np.zeros((len(self.training_dataset.graphs_list), len(new_paths)))

        for i, graph_number in enumerate(graphs_that_contain_selected_column_path):
            for path_number, path in enumerate(paths[i]):
                column_number = new_paths.index(path)
                new_columns[graph_number][column_number] = counts[i][path_number]

        return new_paths, new_columns

    def __get_new_paths(self, selected_path_label, graphs_that_contain_selected_column_path):
        """
        given one path it returns the list of all the possible extension of the input path
        If the path is not present in the graph an empty list is returned for that extension
        """
        if Settings.parallelization is False:
            new_paths = [list(
                self.training_dataset.graphs_list[graph_number].get_new_paths_labels(
                    selected_path_label))
                for graph_number in graphs_that_contain_selected_column_path]
        else:
            pass
        '''
            # ------------------------------------------------------------------------------------------------------------

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
            print(size)
            print(rank)
            if rank == 0:
                split_graphs_list = self.__split(graphs_that_contain_selected_column_path, size)

            graph_number_list = comm.scatter(split_graphs_list, root=0)

            new_paths_for_specific_graph_list = [list(
                self.training_dataset.graphs_list[graph_number].get_new_paths_labels(
                    selected_path_label))
                for graph_number in graph_number_list]

            new_paths = comm.gather(new_paths_for_specific_graph_list, root=0)

            # -----------------------------------------------------------------------------------------------------------
            new_paths = [item for sublist in new_paths for item in sublist]
        '''
        # flattern the list of list
        new_paths = list(set([path for paths_list in new_paths for path in paths_list]))
        return new_paths

    @staticmethod
    def __split(list, n):
        k, m = divmod(len(list), n)
        return (list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    '''
    def __get_new_paths_labels_for_graph_number(self, graph_number, selected_path_label):
        return list(
            self.training_dataset.graphs_list[
                graph_number].get_new_paths_labels(
                selected_path_label))
    '''

    def __initialize_boosting_matrix(self):
        """
        it initializes the attribute boosting_matrix by searching in all the dataset all the metal atoms present in the
         graphs
         Note it handle also the case in which more than in one graph are present multiple metal atoms
        """
        # get a list of all the metal centers atomic numbers
        # metal_centers = list(itertools.chain(*[graph.metal_center for graph in self.dataset.graphs_list]))
        matrix_header = set()
        label_to_graphs = defaultdict(list)

        for i in range(len(self.training_dataset.graphs_list)):
            graph = self.training_dataset.graphs_list[i]
            metal_center_labels = graph.get_metal_center_labels()
            metal_center_labels = [tuple(label) for label in metal_center_labels]
            matrix_header.update(metal_center_labels)
            for label in metal_center_labels:
                label_to_graphs[label].append(int(i))

        boosting_matrix = np.zeros((len(self.training_dataset.graphs_list), len(matrix_header)), dtype=int)
        matrix_header = list(matrix_header)

        for ith_label in range(len(matrix_header)):
            label = matrix_header[ith_label]
            for ith_graph in label_to_graphs[label]:
                graph = self.training_dataset.graphs_list[ith_graph]
                nodes = graph.label_to_node[label[0]]
                boosting_matrix[ith_graph][ith_label] = len(nodes)
                for node in nodes:
                    graph.selected_paths.add_path(path_label=label, path=[node])

        self.boosting_matrix = BoostingMatrix(boosting_matrix, matrix_header)

    def get_boosting_matrix_header(self) -> List[Tuple[int]]:
        return self.boosting_matrix.get_header()

    def get_boosting_matrix(self):
        return self.boosting_matrix

    def get_boosting_matrix_columns_importance_values(self) -> list[float]:
        return self.boosting_matrix.get_columns_importance()

    def get_boosting_matrix_normalized_columns_importance_values(self) -> list[float]:
        return self.boosting_matrix.get_normalized_columns_importance()

    def __expand_boosting_matrix(self, selected_column_number):
        # Following two lines are jut to have mor readable code, everything can be grouped in one line
        length_selected_path = len(self.boosting_matrix.header[selected_column_number])
        path_length_condition = length_selected_path < self.settings.max_path_length

        if not (selected_column_number in self.boosting_matrix.already_selected_columns):
            # if the selected column has never been selected before
            self.boosting_matrix.already_selected_columns.add(selected_column_number)
            if path_length_condition is True:
                selected_column = self.boosting_matrix.matrix[:, selected_column_number]
                selected_path_label = self.boosting_matrix.header[selected_column_number]

                graphs_that_contain_selected_column_path = np.nonzero(selected_column)[0]

                # old version
                # new_paths_labels = self.__get_new_paths(selected_path_label, graphs_that_contain_selected_column_path)
                # new_columns = self.__get_new_columns(new_paths_labels, graphs_that_contain_selected_column_path)

                new_paths_labels, new_columns = self.__get_new_paths_and_columns(selected_path_label,
                                                                                 graphs_that_contain_selected_column_path)

                self.boosting_matrix.add_column(new_columns, new_paths_labels)

    def get_n_iterations(self):
        return self.n_iterations

    def get_selected_paths_in_boosting_matrix(self):
        return self.boosting_matrix.get_selected_paths()

    def get_dataset_dimension(self, dataset: str) -> int:

        if dataset == "train" or dataset == "training":
            if hasattr(self, 'training_dataset'):
                if self.training_dataset is None:
                    return 0
                else:
                    return self.get_dataset("training").get_dimension()
            else:
                return 0

        elif dataset == "test" or dataset == "testing":
            if hasattr(self, 'test_dataset'):
                if self.test_dataset is None:
                    return 0
                else:
                    return self.get_dataset("test").get_dimension()
            else:
                return 0

        else:
            raise TypeError(f"tipe of dataset must be 'train' or 'test', got {dataset} instead")

    def get_dataset(self, dataset: str) -> Dataset:
        '''
        :param dataset: "training" or "test" depending on which of the two we want the observations to come from
        :return: the train/test dataset
        '''

        if dataset == "train" or dataset == "training":
            if hasattr(self, 'training_dataset'):
                if self.training_dataset is not None:

                    return self.training_dataset
                else:
                    return Dataset(graphs_list=None)
            else:
                return Dataset(graphs_list=None)

        elif dataset == "test" or dataset == "testing":
            if hasattr(self, 'test_dataset'):
                if self.test_dataset is not None:
                    return self.test_dataset
                else:
                    return Dataset(graphs_list=None)
            else:
                return Dataset(graphs_list=None)

        else:
            raise TypeError(f"tipe of dataset must be 'train' or 'test', got {dataset} instead")

    def get_patterns_normalized_importance(self) -> Tuple[List[Tuple[int]], List[float]]:
        importance = self.get_boosting_matrix_normalized_columns_importance_values()
        paths = self.get_boosting_matrix_header()
        return paths, importance

    def get_max_path_correlation(self, paths_list: list):
        boosting_dataframe = self.boosting_matrix.get_pandas_matrix()

        highest_correlations = {}
        corr_matrix = boosting_dataframe.corr()
        for col_name in paths_list:
            if col_name in corr_matrix.columns:
                corr_values = corr_matrix[col_name]
                corr_values = corr_values[corr_values.index != col_name]  # exclude self-correlation
                absolute_corr_values = corr_values.abs()  # consider absolute correlation
                highest_corr_coef = np.nanmax(absolute_corr_values)
                highest_correlations[col_name] = highest_corr_coef
        return highest_correlations

    def get_number_of_times_path_has_been_selected(self, path: tuple | int | None = None) -> int | npt.NDArray:
        return self.boosting_matrix.get_number_of_times_path_has_been_selected(path)
