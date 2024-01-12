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

# from pympler import asizeof

if Settings.parallelization is True:
    from mpi4py import MPI


class PatternBoosting:
    def __init__(self, settings=Settings(), model: GradientBoostingModel = None):
        self.settings = settings
        self.model = model
        self.trained = False
        self.test_error = []
        self.train_error = []
        self.average_path_length = []
        self.number_of_learners = []
        self.gradient_boosting_step = GradientBoostingStep()
        self.n_iterations = None
        self.boosting_matrix_matrix_for_test_dataset = None

    def training(self, training_dataset, test_dataset=None):
        """Trains the model, it is possible to call this function multiple times, in this case the dataset used for
        training is always the one took as input the first time the function "training" is called
        In future versions it will be possible to give as input a new dataset"""

        if isinstance(training_dataset, Dataset):
            self.training_dataset = training_dataset

        elif isinstance(training_dataset, list):
            self.training_dataset = Dataset(training_dataset)

        elif training_dataset is None:
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

        else:
            boosting_matrix_matrix = [self.__create_boosting_vector_for_graph(graph) for graph in
                                      training_dataset.graphs_list]
            self.boosting_matrix = BoostingMatrix(boosting_matrix_matrix, self.boosting_matrix.header,
                                                  self.boosting_matrix.columns_importance)

        for iteration_number in range(self.settings.maximum_number_of_steps):
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

            self.train_error.append(self.evaluate(self.training_dataset, self.boosting_matrix.matrix))
            # -------------------------------------------------------------------------------------------------------
            # debug

            # print("Error: ", self.train_error[-1])
            # --------------------------------------------------------------------------------------------------------

            if len(self.train_error) <= 1:
                default_importance_value = np.var(training_dataset.labels)
            else:
                default_importance_value = None
            print("line 94 pattern boosting")
            self.boosting_matrix.update_pattern_importance_of_column(selected_column_number,
                                                                     train_error=self.train_error,
                                                                     default_value=default_importance_value)
            print("line 98 pattern boosting")
            self.number_of_learners.append(iteration_number + 1)

            # expand boosting matrix
            print("------")
            self.__expand_boosting_matrix(selected_column_number)
            print("expanded boosting")

            self.average_path_length.append(self.boosting_matrix.average_path_length())

            if self.train_error[-1] < Settings.target_train_error:
                break

        if test_dataset is not None:
            self.boosting_matrix_matrix_for_test_dataset = boosting_matrix_matrix = self.create_boosting_matrix_for(
                test_dataset)
            self.test_error = self.evaluate_progression(test_dataset, self.boosting_matrix_matrix_for_test_dataset)

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
        if self.trained is False:
            warnings.warn("This model is not trained")
            return None
        if isinstance(graphs_list, GraphPB):
            graphs_list = [graphs_list]
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        if boosting_matrix_matrix is None:
            boosting_matrix_matrix = self.create_boosting_matrix_for(graphs_list)
        prediction = self.model.predict_my(boosting_matrix_matrix)
        return prediction

    def predict_boosting_matrix(self, boosting_matrix_matrix: np.ndarray):
        if self.trained is False:
            warnings.warn("This model is not trained")
            return None
        prediction = self.model.predict_my(boosting_matrix_matrix)
        return prediction

    def create_boosting_matrix_for(self, graphs_list):
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        boosting_matrix_matrix = np.array(
            [self.__create_boosting_vector_for_graph(graph) for graph in graphs_list])
        return boosting_matrix_matrix

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
        boosting_vector = [graph.number_of_time_path_is_present_in_graph(path_label) for path_label in
                           self.boosting_matrix.get_header()]
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
                print("line 266 pattern boosting")

    def get_n_iterations(self):
        return self.n_iterations
