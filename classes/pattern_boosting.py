from classes.graph import GraphPB
from classes.boosting_matrix import BoostingMatrix
from settings import Settings
from classes.gradient_boosting_step import GradientBoostingStep
from classes.dataset import Dataset
from collections import defaultdict
from classes.enumeration.estimation_type import EstimationType
from classes.analysis import Analysis
import numpy as np


class PatternBoosting:
    def __init__(self, settings=Settings(), model=None):
        self.settings = settings
        self.model = model
        self.trained = False
        self.test_error = []
        self.train_error = []
        self.average_path_length = []
        self.number_of_learners = []
        self.analysis = Analysis()
        self.gradient_boosting_step = GradientBoostingStep()

    def training(self, training_dataset, test_dataset=None):
        """Trains the model, it is possible to call this function multiple times, in this case the dataset used for
        training is always the one took as input the first time the function "training" is called
        In future versions it will be possible to give as input a new dataset"""

        if isinstance(training_dataset, Dataset):
            self.training_dataset = training_dataset

        elif isinstance(training_dataset, list):
            self.training_dataset = Dataset(training_dataset)
        else:
            raise TypeError("Input dataset not recognized")

        if test_dataset is not None:
            if isinstance(test_dataset, Dataset):
                self.test_dataset = test_dataset

            elif isinstance(test_dataset, list):
                self.test_dataset = Dataset(test_dataset)
            else:
                raise TypeError("Input test dataset not recognized")

        # if it is the first time we train this model
        if self.trained is False:
            self.trained = True
            self.__initialize_boosting_matrix()

        else:
            boosting_matrix_matrix = [self.__create_boosting_vector_for_graph(graph) for graph in
                                      training_dataset.graphs_list]
            self.boosting_matrix = BoostingMatrix(boosting_matrix_matrix, self.boosting_matrix.header,
                                                  self.boosting_matrix.patterns_importance)

        for iteration_number in range(self.settings.maximum_number_of_steps):
            print("Step number ", iteration_number + 1)

            selected_column_number, self.model = self.gradient_boosting_step.select_column(model=self.model,
                                                                                           boosting_matrix=self.boosting_matrix,
                                                                                           labels=self.training_dataset.labels,
                                                                                           number_of_learners=iteration_number + 1)
            print("column selected")

            if test_dataset is not None:
                self.test_error.append(self.evaluate(self.test_dataset))
            self.train_error.append(self.evaluate(self.training_dataset))

            if len(self.train_error) <= 1:
                default_importance_value = np.var(training_dataset.labels)
            else:
                default_importance_value = None
            self.boosting_matrix.update_pattern_importance_of_column(selected_column_number,
                                                                     train_error=self.train_error,
                                                                     default_value=default_importance_value)

            self.number_of_learners.append(iteration_number + 1)

            # expand boosting matrix

            self.__expand_boosting_matrix(selected_column_number)

            self.average_path_length.append(self.boosting_matrix.average_path_length())

        # -------------------------------------------------------------------------------------------------------------
        # error plots
        # cut first point

        cut_point = 1
        if Settings.estimation_type == EstimationType.regression:
            self.analysis.plot_informations(self.number_of_learners[cut_point:], self.train_error[cut_point:],
                                            tittle="train error",
                                            x_label="number of learners",
                                            y_label="MSE")

            if test_dataset is not None:
                self.analysis.plot_informations(self.number_of_learners[cut_point:], self.test_error[cut_point:],
                                                tittle="test error",
                                                x_label="number of learners",
                                                y_label="MSE")
        elif Settings.estimation_type == EstimationType.classification:
            self.analysis.plot_informations(self.number_of_learners, self.train_error, tittle="train classification",
                                            x_label="number of learners", y_label="jaccard score")

            if test_dataset is not None:
                self.analysis.plot_informations(self.number_of_learners, self.test_error, tittle="test classification",
                                                x_label="number of learners", y_label="jaccard score")

        # -----------------------------------------------------------
        # average_path_length_plot
        self.analysis.plot_informations(self.number_of_learners, self.average_path_length, tittle="Average path length",
                                        x_label="number of learners", y_label="average path length")

        self.analysis.print_performance_information(self.boosting_matrix, self.train_error, self.test_error)
        self.analysis.print_test_dataset_info(test_dataset)
        self.analysis.analyse_path_length_distribution(self.boosting_matrix)
        # -----------------------------------------------------------

    def predict(self, dataset: Dataset):
        boosting_matrix_matrix = [self.__create_boosting_vector_for_graph(graph) for graph in dataset.graphs_list]
        prediction = self.model.predict_my(boosting_matrix_matrix)
        return prediction

    def evaluate(self, dataset: Dataset):

        boosting_matrix_matrix = np.array(
            [self.__create_boosting_vector_for_graph(graph) for graph in dataset.graphs_list])
        error = self.model.evaluate(boosting_matrix_matrix, dataset.labels)
        return error

    def __create_boosting_vector_for_graph(self, graph: GraphPB) -> np.array:
        boosting_vector = [graph.number_of_time_path_is_present_in_graph(path_label) for path_label in
                           self.boosting_matrix.header]
        return np.array(boosting_vector)

    def __get_new_columns(self, new_paths, graphs_that_contain_selected_column_path):
        """
        given a st of paths and a set of graphs that contains the given paths it returns the new column that should
        be added to the dataset. in each line of the column the value represent the number of times the path that
        correspond to the column is present in the graph.
        The order of the columns follows the order of the input vector of paths
        """

        new_columns = np.zeros((len(self.training_dataset.graphs_list), len(new_paths)))

        for path_number in range(len(new_paths)):
            path = new_paths[path_number]
            for graph_number in graphs_that_contain_selected_column_path:
                graph = self.training_dataset.graphs_list[graph_number]
                n = graph.number_of_times_selected_path_is_present(path)
                new_columns[graph_number][path_number] = n

        return new_columns

    def __get_new_paths(self, selected_path_label, graphs_that_contain_selected_column_path):
        """
        given one path it returns the list of all the possible extension of the input path
        If the path is not present in the graph an empty list is returned for that extension
        """

        new_paths = [list(
            self.training_dataset.graphs_list[graph_number].get_new_paths_labels_and_add_them_to_the_dictionary(
                selected_path_label))
            for graph_number in graphs_that_contain_selected_column_path]
        new_paths = list(set([path for paths_list in new_paths for path in paths_list]))
        return new_paths

    def __initialize_boosting_matrix(self):
        """
        it initialize the attribute boosting_matrix by searching in all the dataset all the metal atoms present in the
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

        if (not (selected_column_number in self.boosting_matrix.already_selected_columns)) and path_length_condition:
            # if the selected column has never been selected before
            self.boosting_matrix.already_selected_columns.add(selected_column_number)

            selected_column = self.boosting_matrix.matrix[:, selected_column_number]
            selected_path_label = self.boosting_matrix.header[selected_column_number]

            graphs_that_contain_selected_column_path = np.nonzero(selected_column)[0]

            new_paths_labels = self.__get_new_paths(selected_path_label, graphs_that_contain_selected_column_path)
            new_columns = self.__get_new_columns(new_paths_labels, graphs_that_contain_selected_column_path)

            self.boosting_matrix.add_column(new_columns, new_paths_labels)
