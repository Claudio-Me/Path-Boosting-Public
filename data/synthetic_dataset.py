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


class SyntheticDataset:
    '''
    It takes the given dataset and generates new labels from the formula y=c0p0 + c1p1 + c2p2 + ...
    Where p0...pn are the number of times target_path[0]...target_path[n] are present in the selected graph
    '''

    def __init__(self):
        self.target_paths = list({(57, 7, 7), (57, 7), (57,),
                                (72, 7, 14), (72, 7), (72,),
                                (78, 6, 7), (78, 6), (78,),
                                (47, 7, 7), (47, 7), (47,),
                                (74, 15, 8), (74, 15), (74,),
                                (80, 7, 7), (80, 7), (80,),
                                (77, 7, 7), (77, 7), (77,),
                                (40, 7, 14), (40, 7), (40,),
                                (21, 7, 14), (21, 7), (21,),
                                (27, 6, 5), (27, 6), (27,),
                                (27, 6, 8),
                                (42, 7, 7), (42, 7), (42,),
                                (39, 7, 7), (39, 7), (39,),
                                (39, 7, 14),
                                (39, 6, 7), (39, 6),
                                (39, 6, 14),
                                (39, 6, 5),
                                (45, 7, 7), (45, 7), (45,),
                                (48, 8, 7), (48, 8), (48,)})

        self.variance = 1
        self.coefficients = np.random.uniform(2, 3, len(self.target_paths))
        self.keep_probability = 0.01
        self.new_graphs_list = []
        self.new_labels_list = []
        self.number_paths_counting = None

        simple = list({(28,), (28, 7), (28, 7, 6)})
        a = list({(28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6), (28, 7, 6, 6), (28, 7, 6)})

        b = [(47, 7), (42, 7, 6), (75, 8), (29,), (75, 17), (47,), (79, 17), (40,), (28, 7),  # all connected
             (28, 16), (46, 6), (23, 7, 6), (28, 7, 7), (75, 7, 6), (75, 7), (75, 16),
             (79, 7), (46, 6, 7), (79, 16), (23, 8), (42, 6), (42, 15), (42, 8), (28, 7, 6),
             (30, 8), (30, 17), (75, 6), (42, 7, 7), (79, 6), (79, 15), (23, 7), (78, 16),
             (28, 6), (28, 15), (28, 16, 6), (28, 8), (42, 7), (28, 17),
             (28, 35), (30, 7), (79, 7, 6), (30, 16), (75, 15), (46, 6, 6), (75,), (77,), (27,), (22,), (30,),
             (23,), (24,), (79,), (74,), (28,), (46,), (73,), (45,), (48,),
             (42,), (26,), (44,), (25,), (78,), (80,)]

        b2 = [(42, 7, 6), (29,), (47,), (40,), (23, 7, 6), (28, 7, 7), (75, 7, 6), (46, 6, 7),
              (28, 7, 6), (42, 7, 7), (28, 16, 6), (79, 7, 6), (46, 6, 6), (75,),
              (77,), (27,), (22,), (30,), (23,), (24,), (79,), (74,), (28,), (46,), (73,), (45,), (48,), (42,), (26,),
              (44,), (25,), (78,), (80,)]

        c = [
            (28,), (28, 7),  # only variations of
            (28, 7, 16), (28, 7, 6), (28, 7, 8), (28, 7, 14), (28, 7, 7), (28, 7, 6, 6), (28, 7, 6, 15),
            (28, 7, 6, 8), (28, 7, 6, 7), (28, 7, 6, 16), (28, 7, 6, 6, 9), (28, 7, 6, 6, 6), (28, 7, 6, 6, 15),
            (28, 7, 6, 6, 8), (28, 7, 6, 6, 17), (28, 7, 6, 6, 7), (28, 7, 6, 6, 16), (28, 7, 6, 6, 35),
            (28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6, 9), (28, 7, 6, 6, 6, 15), (28, 7, 6, 6, 6, 8), (28, 7, 6, 6, 6, 17),
            (28, 7, 6, 6, 6, 7), (28, 7, 6, 6, 6, 16), (28, 7, 6, 6, 6, 6), (28, 7, 7, 16), (28, 7, 7, 5),
            (28, 7, 7, 6), (28, 7, 7, 7), (28, 7), (28, 7, 8), (28, 7, 16), (28, 7, 6), (28, 7, 7), (28, 7, 6, 6),
            (28, 7, 6, 15), (28, 7, 6, 8), (28, 7, 6, 7), (28, 7, 6, 16), (28, 7, 6, 6, 9), (28, 7, 6, 6, 6),
            (28, 7, 6, 6, 15), (28, 7, 6, 6, 8), (28, 7, 6, 6, 17), (28, 7, 6, 6, 7), (28, 7, 6, 6, 16),
            (28, 7, 6, 6, 35), (28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6, 9), (28, 7, 6, 6, 6, 15), (28, 7, 6, 6, 6, 8),
            (28, 7, 6, 6, 6, 17), (28, 7, 6, 6, 6, 7), (28, 7, 6, 6, 6, 16), (28, 7, 6, 6, 6, 6), (28, 7, 7, 16),
            (28, 7, 7, 5), (28, 7, 7, 6), (28, 7, 7, 7),

        ]

        right_ratio = [(57, 7, 7),
                       (72, 7, 14),
                       (78, 6, 7),
                       (47, 7, 7),
                       (74, 15, 8),
                       (80, 7, 7),
                       (77, 7, 7),
                       (40, 7, 14),
                       (21, 7, 14),
                       (27, 6, 5),
                       (27, 6, 8),
                       (42, 7, 7),
                       (39, 7, 7),
                       (39, 7, 14),
                       (39, 6, 7),
                       (39, 6, 14),
                       (39, 6, 5),
                       (45, 7, 7),
                       (48, 8, 7)]
        right_ratio_expanded = [(57, 7, 7), (57, 7), (57,),
                                (72, 7, 14), (72, 7), (72,),
                                (78, 6, 7), (78, 6), (78,),
                                (47, 7, 7), (47, 7), (47,),
                                (74, 15, 8), (74, 15), (74,),
                                (80, 7, 7), (80, 7), (80,),
                                (77, 7, 7), (77, 7), (77,),
                                (40, 7, 14), (40, 7), (40,),
                                (21, 7, 14), (21, 7), (21,),
                                (27, 6, 5), (27, 6), (27),
                                (27, 6, 8),
                                (42, 7, 7), (42, 7), (42,),
                                (39, 7, 7), (39, 7), (39,),
                                (39, 7, 14),
                                (39, 6, 7), (39, 6),
                                (39, 6, 14),
                                (39, 6, 5),
                                (45, 7, 7), (45, 7), (45,),
                                (48, 8, 7), (48, 8), (48,)]

    def create_dataset_from_5k_selection_graph(self, save_on_file=True, filename: str = "5_k_selection_graphs",
                                               new_file_name="5k_synthetic_dataset"):
        dataset = load_dataset_from_binary(filename=filename)
        # each row is a different graph, each column is a different path
        self.number_paths_counting = np.array(
            [[graph.number_of_time_path_is_present_in_graph(path) for path in self.target_paths] for
             graph in dataset.graphs_list])
        print("line 130 synthetic_dataset")
        new_labels = self.__formula_new_labels(self.number_paths_counting)
        print("line 1132 synthetic_dataset")
        dataset.labels = list(new_labels)
        print("line 134 synthetic_dataset")
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
        new_dataset = classes.dataset.Dataset(graphs_list=self.new_graphs_list, labels=self.new_labels_list)
        if save_on_file is True:
            save_dataset_in_binary_file(new_dataset, filename=new_file_name)

        return new_dataset

    def __formula_new_labels(self, number_paths_counting, add_noise=True):

        assert len(self.coefficients) == len(self.target_paths)

        # ------------------------------------------------------------------------------------------
        # can be rewritten using vector multiplication, but I don't have internet now to check how to do it
        # also if number_of_paths_counting is a matrix we can use matrix multiplications
        number_paths_counting = np.array(number_paths_counting)
        y = np.matmul(number_paths_counting, self.coefficients)

        # add random noise
        if add_noise is True:
            noise = np.random.normal(0, self.variance, len(y))
            y = y + noise

        # ------------------------------------------------------------------------------------------
        return y

    def oracle_model_evaluate(self, graphs_list, labels):  # graphs_list: list[GraphPB]
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

    def oracle_model_predict(self, graphs_list):  # graphs_list: list[GraphPB]
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
                                    matrix.get_selected_paths()]
        data = {
            "Selected Path": matrix.get_selected_paths(),
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
            table_synthetic_dataset_results = self.get_latex_code_for(table_synthetic_dataset_results)

            table_all_selected_paths = self.get_number_of_times_all_path_are_selected(
                pattern_boosting=pattern_boosting_model)
            if show is True:
                print(table_all_selected_paths)
            table_all_selected_paths = self.get_latex_code_for(table_all_selected_paths)

        dataset_info_table = self.get_table_dataset_info(pattern_boosting_model)
        if show is True:
            print(dataset_info_table)
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

    def table_synthetic_dataset_results(self, pattern_boosting: PatternBoosting):
        assert (pattern_boosting is not None)
        boosting_matrix = pattern_boosting.boosting_matrix
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
    def generate_random_graph(dimension):
        graph = nx.gaussian_random_partition_graph(n=dimension, s=dimension, v=dimension, p_in=0.4, p_out=1)
        graph = GraphPB.from_GraphNX_to_GraphPB(graph)

        return graph
