import random
import sys

import numpy as np
import pandas as pd

import networkx as nx

import classes.dataset
from data.data_reader import load_dataset_from_binary, save_dataset_in_binary_file, get_save_location
from classes.pattern_boosting import PatternBoosting
from classes.graph import GraphPB
from settings import Settings
from sklearn import metrics
from pathlib import Path

class SyntheticDataset:
    '''
    It takes the given dataset and generates new labels from the formula y=c0p0 + c1p1 + c2p2 + ...
    Where p0...pn are the number of times target_path[0]...target_path[n] are present in the selected graph
    '''

    def __init__(self, settings: Settings):



        self.target_paths = Settings.target_paths

        self.variance = Settings.noise_variance
        random_generator_for_coefficients = random.Random()
        random_generator_for_coefficients.seed(Settings.random_coefficients_synthetic_dataset_seed)
        self.coefficients = [random_generator_for_coefficients.uniform(2, 3) for _ in range(len(self.target_paths))]

        for i, coefficient in enumerate(self.coefficients):
            # self.coefficients[i] = coefficient * pow(10, 2 - len(self.target_paths[i]))
            self.coefficients[i] = coefficient / len(self.target_paths[i])

        self.keep_probability = 0.0
        self.new_graphs_list: list[GraphPB] = []
        self.new_labels_list = []
        self.number_paths_counting = None

    def create_dataset_from_5k_selection_graph(self, save_on_file=True, filename: str = "5_k_selection_graphs",
                                               new_file_name="5k_synthetic_dataset", directory=None):
        dataset = load_dataset_from_binary(directory=directory, filename=filename)
        # each row is a different graph, each column is a different path
        self.number_paths_counting = np.array(
            [[graph.number_of_time_path_is_present_in_graph(path) for path in self.target_paths] for
             graph in dataset.graphs_list])
        new_labels = self.__formula_new_labels(self.number_paths_counting, add_noise=True)

        dataset.labels = list(new_labels)
        a = self.number_paths_counting.sum(axis=1)
        self.number_of_graphs_that_contains_target_path = np.count_nonzero(a)

        for i in range(len(a)):
            if a[i] != 0:
                self.new_graphs_list.append(dataset.graphs_list[i])
                self.new_labels_list.append(new_labels[i])
            else:
                if random.uniform(0, 1) < self.keep_probability:
                    self.new_graphs_list.append(dataset.graphs_list[i])
                    self.new_labels_list.append(new_labels[i])

        for i, graph in enumerate(self.new_graphs_list):
            graph.set_label_value(self.new_labels_list[i])
        new_dataset = classes.dataset.Dataset(graphs_list=self.new_graphs_list, labels=self.new_labels_list)

        if save_on_file is True:
            current_dir = Path(__file__).parent.resolve()
            save_dataset_in_binary_file(new_dataset, directory=current_dir, filename=new_file_name)

        return new_dataset

    def __formula_new_labels(self, number_paths_counting, add_noise=True):

        assert len(self.coefficients) == len(self.target_paths)

        # ------------------------------------------------------------------------------------------
        # can be rewritten using vector multiplication, but I don't have internet now to check how to do it
        # also if number_of_paths_counting is a matrix we can use matrix multiplications
        number_paths_counting = np.array(number_paths_counting)
        # for some reason this line does not work on the server
        # y = np.matmul(number_paths_counting, self.coefficients)
        # y = number_paths_counting @ self.coefficients
        y = np.array([sum(a * b for a, b in zip(A_row, self.coefficients)) for A_row in number_paths_counting])

        # add random noise
        if add_noise is True:
            # noise = np.random.normal(0, self.variance, len(y))
            noise = [Settings.random_generator_for_noise_in_synthetic_dataset.normalvariate(0, np.sqrt(self.variance)) for _ in range(len(y))]

            y = y + noise

        # ------------------------------------------------------------------------------------------
        return y

    def oracle_model_evaluate(self, graphs_list: list[GraphPB] , labels):  # graphs_list: list[GraphPB]
        if not hasattr(graphs_list, '__iter__'):
            # if graph_list is not a list then I assume it is a singe graph
            graphs_list = [graphs_list]
            labels = [labels]

        y_pred = self.oracle_model_predict(graphs_list)
        if Settings.final_evaluation_error == "MSE":
            model_error = metrics.mean_squared_error(labels, y_pred)
        elif Settings.final_evaluation_error == "absolute_mean_error":
            model_error = metrics.mean_absolute_error(labels, y_pred)
        else:
            raise ValueError("measure error not found")
        return model_error

    def oracle_model_predict(self, graphs_list: list[GraphPB]):  # graphs_list: list[GraphPB]
        if not hasattr(graphs_list, '__iter__'):
            # if graph_list is not a list then I assume it is a singe graph
            graphs_list = [graphs_list]

            # each row is a different graph, each column is a different path
        graphs_counting_target_paths = np.array(
            [[graph.number_of_time_path_is_present_in_graph(path) for path in self.target_paths] for graph in
             graphs_list])

        predicted_labels = self.__formula_new_labels(graphs_counting_target_paths, add_noise=False)
        return predicted_labels

    def get_number_of_times_all_path_are_selected(self, pattern_boosting: PatternBoosting) -> pd.DataFrame:
        matrix = pattern_boosting.boosting_matrix
        n_times_path_is_selected = [matrix.get_number_of_times_path_has_been_selected(path) for path in
                                    matrix.get_header()]
        data = {
            "Selected Path": matrix.get_header(),
            "n times has been selected": n_times_path_is_selected
        }
        return pd.DataFrame(data)

    def get_target_paths_table(self, pattern_boosting: PatternBoosting = None) -> pd.DataFrame:
        # it creates a pandas dataset containing all the info
        total_number_of_times_a_path_is_present = self.number_paths_counting.sum(axis=0)
        number_of_graph_is_present = np.count_nonzero(self.number_paths_counting, axis=0)
        if pattern_boosting is None:
            data = {"Path": self.target_paths,
                    "Real Coeff": self.coefficients,
                    "Times is present": total_number_of_times_a_path_is_present,
                    "Number of graph is present": number_of_graph_is_present
                    }
        else:
            est_coeff = self.get_estimated_coefficients(pattern_boosting)
            patterns_importance = self.get_patterns_importance(pattern_boosting)
            times_selected = self.get_times_paths_are_selected(pattern_boosting)
            data = {"Path": self.target_paths,
                    "Real Coeff": self.coefficients,
                    "Est Coeff": est_coeff,
                    "Importance": patterns_importance,
                    "Times selected": times_selected,
                    "N° is present": total_number_of_times_a_path_is_present,
                    "Graphs present in": number_of_graph_is_present
                    }
        table = pd.DataFrame(data)
        if not (pattern_boosting is None):
            table = table.sort_values(by=['Importance'], ascending=False)
        return table

    def get_times_paths_are_selected(self, pattern_boosting: PatternBoosting) -> list:
        matrix = pattern_boosting.boosting_matrix
        return [matrix.get_number_of_times_path_has_been_selected(path) for path in self.target_paths]

    def get_patterns_importance(self, pattern_boosting: PatternBoosting):
        matrix = pattern_boosting.boosting_matrix
        return [matrix.get_importance_of(path) for path in self.target_paths]

    def get_table_dataset_info(self, pattern_boosting: PatternBoosting) -> pd.DataFrame:

        oracle_error = self.oracle_model_evaluate(graphs_list=pattern_boosting.test_dataset.get_graphs_list(),
                                                  labels=pattern_boosting.test_dataset.get_labels())
        data = {"Mean y": [np.mean(self.new_labels_list)],
                "Number of observations": [len(self.new_labels_list)],
                "Obs containing target path": [self.number_of_graphs_that_contains_target_path],
                "Oracle error": [oracle_error]
                }

        table = pd.DataFrame(data)
        return table

    def get_latex_code_for(self, table: pd.DataFrame) -> str:
        assert isinstance(table, pd.DataFrame)
        n_cols = len(table.axes[1])
        column_format = ["|l"] * n_cols
        column_format = "".join(column_format)
        column_format = column_format + '|'
        rounded_table = table.round(2)
        return rounded_table.style.hide(axis="index").to_latex(
            clines="all;index",
            column_format=column_format,
            hrules=True
        )

    def create_and_save_tables_in_latex(self, pattern_boosting_model: PatternBoosting = None,
                                        writing_method: str = 'w', show=True, save=True):

        target_paths_table = self.get_target_paths_table(pattern_boosting_model)
        if show is True:
            print(target_paths_table)
        target_paths_table = self.get_latex_code_for(target_paths_table)
        if pattern_boosting_model is not None:
            table_synthetic_dataset_results = self.table_synthetic_dataset_results(pattern_boosting_model)
            if show is True:
                print(table_synthetic_dataset_results)
            if save is True:
                saving_location = get_save_location("table_synthetic_dataset_results", '.tex')
                table_synthetic_dataset_results.to_latex(saving_location, index=False)

            table_synthetic_dataset_results = self.get_latex_code_for(table_synthetic_dataset_results)

            table_all_selected_paths = self.get_number_of_times_all_path_are_selected(
                pattern_boosting=pattern_boosting_model)
            if show is True:
                print(table_all_selected_paths)

            if save is True:
                saving_location = get_save_location("table_all_selected_paths", '.tex')
                table_all_selected_paths.to_latex(saving_location, index=False)

            table_all_selected_paths = self.get_latex_code_for(table_all_selected_paths)

        dataset_info_table = self.get_table_dataset_info(pattern_boosting_model)
        if show is True:
            print(dataset_info_table)

        if save is True:
            saving_location = get_save_location("dataset_info_table", '.tex')
            dataset_info_table.to_latex(saving_location, index=False)

        dataset_info_table = self.get_latex_code_for(dataset_info_table)

        string = target_paths_table + "\n\n"

        if pattern_boosting_model is not None:
            string = string + table_synthetic_dataset_results + "\n\n"
            string = string + table_all_selected_paths + "\n\n"

        string = string + dataset_info_table + "\n\n"

        saving_location = get_save_location("Synthetic_dataset_info", '.txt')
        original_stdout = sys.stdout  # Save a reference to the original standard output
        if save is True:
            with open(saving_location,
                      writing_method) as f:  # change 'w' with 'a' to append text at the end of the file
                sys.stdout = f  # Change the standard output to the file we created.

                print(string)

                sys.stdout = original_stdout  # Reset the standard output to its original value

    def table_synthetic_dataset_results(self, pattern_boosting: PatternBoosting) -> pd.DataFrame:
        assert (pattern_boosting is not None)
        boosting_matrix = pattern_boosting.get_boosting_matrix()
        selected_paths = []
        for selected_column in boosting_matrix.already_selected_columns:
            selected_paths.append(boosting_matrix.header[selected_column])

        # count the number of really important paths that have been selected by the algorithm
        counter = 0

        for target_path in set(self.target_paths):
            if target_path in selected_paths:
                counter += 1
        print("Total number of target paths: ", len(self.target_paths))

        data = {"Target paths spotted": [counter],
                "selected paths": [len(selected_paths)],
                "Steps": [pattern_boosting.get_n_iterations()],
                "train err": [pattern_boosting.train_error[-1]]
                }
        if pattern_boosting.test_error is not None:
            data["test err"] = [pattern_boosting.test_error[-1]]
        return pd.DataFrame(data)

    def get_estimated_coefficients(self, pattern_boosting: PatternBoosting):
        '''
        It computes the estimated coefficients of the target paths by using a boosting matrix that is 1 oly in the target path entry.
        Note, if a "specific target_path" it is not present in the boosting matrix header, it returns "None" fot that entry
        '''
        matrix = self.__create_boosting_matrix_with_target_paths(pattern_boosting.boosting_matrix.get_header())
        estimated_coefficients = pattern_boosting.model.predict_my(boosting_matrix_matrix=matrix)
        for i in range(len(self.target_paths)):
            if not (self.target_paths[i] in pattern_boosting.boosting_matrix.get_header()):
                estimated_coefficients[i] = None
        return estimated_coefficients

    def __create_boosting_matrix_with_target_paths(self, header: list):
        matrix = np.zeros((len(self.target_paths), len(header)), dtype=np.int8)
        for i in range(len(self.target_paths)):
            if self.target_paths[i] in header:
                checking_path = self.target_paths[i]
                for j in range(len(self.target_paths[i]) - 1):
                    last_index = len(self.target_paths[i]) - (j + 1)

                    path = checking_path[:last_index]
                    column = header.index(path)
                    matrix[i, column] = 1

        return matrix

    @staticmethod
    def generate_random_graph(dimension, settings):
        graph = nx.gaussian_random_partition_graph(n=dimension, s=dimension, v=dimension, p_in=0.4, p_out=1)
        graph = GraphPB.from_GraphNX_to_GraphPB(nx_Graph=graph, settings=settings)

        return graph


