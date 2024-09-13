import numpy as np
import pandas as pd
from PyAstronomy import pyasl
from typing import List, Tuple

from pandas.core.interchange import dataframe

from classes.boosting_matrix import BoostingMatrix
import networkx as nx
from data import data_reader
from classes.graph import GraphPB
from collections import defaultdict
import matplotlib.pyplot as plt
import ast
from settings import Settings
import copy


class ExtendedBoostingMatrix:

    def __init__(self, df: pd.DataFrame | None = None):
        self.df: pd.DataFrame | None = df
        if self.df is not None:
            ExtendedBoostingMatrix.sort_df_columns(self.df)

    def create_extend_boosting_matrix(self, selected_paths: list[tuple],
                                      list_graphs_nx: list[nx.classes.multigraph.MultiGraph], convert_to_sparse=False):
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset

        self.df = ExtendedBoostingMatrix.create_extend_boosting_matrix_for(selected_paths, list_graphs_nx,
                                                                           convert_to_sparse)

    @staticmethod
    def __get_all_possible_attributes(dataset: list[nx.classes.multigraph.MultiGraph]) -> set[str]:
        unique_attributes = set()
        for graph in dataset:
            nodes_attributes_list = copy.deepcopy(graph.nodes(data=True))
            for node, attributes in nodes_attributes_list:
                unique_attributes.update(attributes.keys())
        if 'node_position' in unique_attributes:
            unique_attributes.remove('node_position')
        if 'node_label' in unique_attributes:
            unique_attributes.remove('node_label')
        # add "n_times_present", it refers to the attribute "(path)_times_present"
        unique_attributes.add('n_times_present')
        return unique_attributes

    @staticmethod
    # finds all the possible attributes of the node
    def __get_node_attributes_of_nx_graph(graph, node_id) -> dict | None:
        if str(node_id) in graph.nodes:
            return copy.deepcopy(graph.nodes[str(node_id)])
        else:
            return None

    @staticmethod
    def __get_all_possible_node_attributes_in_the_dataset(dataset: list[nx.classes.multigraph.MultiGraph],
                                                          selected_paths: list[tuple]) -> list[str]:
        # it returns all the possible combination between path and attributes
        unique_attributes = ExtendedBoostingMatrix.__get_all_possible_attributes(dataset)

        columns_name = set()
        for path in selected_paths:
            # add the underscore to omogenize the input, otherwise when split('_',1) is called (it is done in another method) it returns only one argument
            for attribute in sorted(unique_attributes):
                columns_name.add(ExtendedBoostingMatrix.__get_column_name(path, attribute))

        return list(columns_name)

    @staticmethod
    def __get_column_name(path: tuple, attribute: str):
        return str(path) + '_' + attribute

    def plot_sparsity_matrix(self, save_fig=False):
        return self.__plot_sparsity_matrix(self.df, save_fig)

    @staticmethod
    def __plot_sparsity_matrix(df: pd.DataFrame, save_fig=False):
        # Get the sparsity pattern for each SparseDtype column
        locations = []
        for column in df.columns:
            sparse_series = df[column]
            if pd.api.types.is_sparse(sparse_series):
                sparse_array = sparse_series.array
                # Get the indices of the non-zero entries
                non_zero_indices = sparse_array.sp_index.to_int_index().indices
                col_index = df.columns.get_loc(column)
                locations.extend(zip(non_zero_indices, [col_index] * len(non_zero_indices)))

        # Unpack the locations to separate lists of rows and columns
        rows, cols = zip(*locations)

        plt.figure(figsize=(10, 6))
        plt.scatter(cols, rows, alpha=0.5, s=0.01)
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.gca().invert_yaxis()  # Invert the y-axis to match the matrix representation
        if save_fig is True:
            plt.savefig()
        plt.show()

    @staticmethod
    def __column_sort_key(column_name):
        """
        Generate a sorting key for a DataFrame column based on a composite column name.

        The column name is expected to contain a string representation of a tuple of integers,
        followed by an underscore and another arbitrary name. The sorting key is based on the
        length of the tuple and the values within it.

        Parameters:
        column_name (str): The name of the DataFrame column to generate a key for.

        Returns:
        tuple: A tuple where the first element is the length of the tuple and the second element
               is the tuple of integers itself, which will be used for sorting.

        Example:
        >>> ExtendedBoostingMatrix.__column_sort_key("(1,2)_name")
        (2, (1, 2))

        """
        # Split the column name on the underscore to extract the string-tuple and the name
        if column_name == 'target':
            return 0, (0,)
        string_tuple, feature_name = column_name.split('_', 1)

        # Convert the string-tuple to an actual tuple
        tuple_of_ints = ast.literal_eval(string_tuple)
        # Create a sorting key: (length of the tuple, the tuple itself)
        return (len(tuple_of_ints), tuple_of_ints, str(feature_name))

    @staticmethod
    def sort_df_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        sorted_columns = sorted(dataframe.columns, key=ExtendedBoostingMatrix.__column_sort_key)

        dataframe = dataframe.reindex(columns=sorted_columns)
        return dataframe

    @staticmethod
    def __parse_tuple_from_colname(column_name):
        """
        Extract the tuple portion from a DataFrame column name.

        The column name is expected to start with a string representation of a tuple,
        followed by an underscore and then an arbitrary suffix.

        Parameters:
        column_name (str): The column name from which to extract the tuple.

        Returns:
        tuple: The tuple extracted from the column name, as actual integers.

        Example:
        >>> __parse_tuple_from_colname("(1,2,)_some_name")
        (1, 2,)
        """
        if column_name == 'target':
            return (0,)

        string_tuple, _ = column_name.split('_', 1)
        tuple_of_ints = ast.literal_eval(string_tuple)
        return tuple_of_ints

    def associate_paths_to_columns(self, selected_paths):
        """
        Create a dictionary associating tuples in `selected_paths` with DataFrame column indices.

        For each tuple in `selected_paths`, this function finds DataFrame columns whose names
        start with a string representation of a sub-tuple that matches the beginning of the tuple.
        It returns a dictionary where each key is a tuple from `selected_paths`, and the value
        is a list of column indices where a matching sub-tuple is found.

        Parameters:
        df (pd.DataFrame): The DataFrame with columns to be associated with tuples in `selected_paths`.
        selected_paths (list of tuple): A list of tuples for which to find matching DataFrame columns.

        Returns:
        dict: A dictionary mapping each tuple in `selected_paths` to a list of DataFrame column indices.

        Example:
        df_example = pd.DataFrame({
            "(1,2,)_some_name": [1, 2, 3],
            "(1,)_another_name": [4, 5, 6],
            "(2,3,)_different_name": [7, 8, 9],
            "(1,2,3,4)_name": [10, 11, 12]
        })
        selected_paths_example = [(1, 2, 4, 5), (1, 1, 2, 4, 5)]
        >>> associate_paths_to_columns(df_example, selected_paths_example)
        {(1, 2, 4, 5): [0, 3], (1, 1, 2, 4, 5): []}
        """
        tuple_to_column_indices = {}

        # Iterating through selected paths and DataFrame columns to build the dictionary
        for path in selected_paths:
            tuple_to_column_indices[path] = []
            for i, col in enumerate(self.df.columns):
                # Parse the tuple part of the column name
                column_tuple = self.__parse_tuple_from_colname(col)
                # Check if the column tuple is a sub-tuple at the beginning of the current path
                if path[:len(column_tuple)] == column_tuple:
                    tuple_to_column_indices[path].append(i)

        return tuple_to_column_indices

    def get_pandas_dataframe(self) -> pd.DataFrame:
        return self.df

    @staticmethod
    def get_features_interaction_constraints(selected_paths: list[tuple], all_attributes: list[str] | None = None,
                                             list_graphs_nx: list[nx.classes.multigraph.MultiGraph] | None = None):
        # it returns a dictionary where to each labelled path is associated a list containing the name of the columns that contains features relative to said path
        if all_attributes is None:
            all_attributes = ExtendedBoostingMatrix.__get_all_possible_attributes(list_graphs_nx)

        dict_of_interaction_constraints = dict()
        for labelled_path in selected_paths:
            columns_names = []
            for attribute in all_attributes:

                for i in range(len(labelled_path), 0, -1):
                    sub_tuple = labelled_path[:i]
                    columns_names.append(ExtendedBoostingMatrix.__get_column_name(sub_tuple, attribute))
            dict_of_interaction_constraints[labelled_path] = columns_names

        return dict_of_interaction_constraints

    @staticmethod
    def create_boosting_matrix(selected_paths: list[tuple],
                               list_graphs_nx: list[nx.classes.multigraph.MultiGraph],
                               ebm_dataframe: pd.DataFrame = None) -> pd.DataFrame:
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)

        if ebm_dataframe is not None:
            column_names = [str(path) + '_' + "n_times_present" for path in selected_paths]
            boosting_matrix_df = ebm_dataframe[column_names]
            boosting_matrix_df.columns = selected_paths
            return boosting_matrix_df


        else:
            # function to help the retrival of attributes in nx graphs
            graphsPB_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in list_graphs_nx]
            dictionary_for_dataframe = defaultdict(list)

            for path in selected_paths:
                for graph in graphsPB_list:
                    dictionary_for_dataframe[path].append(
                        graph.number_of_time_path_is_present_in_graph(path_label=path))

            boosting_matrix_df = pd.DataFrame(dictionary_for_dataframe, dtype='int')
            boosting_matrix_df.columns = selected_paths
            return boosting_matrix_df

    @staticmethod
    def create_extend_boosting_matrix_for(selected_paths: list[tuple],
                                          list_graphs_nx: list[nx.classes.multigraph.MultiGraph],
                                          convert_to_sparse=False) -> pd.DataFrame:
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)

        # function to help the retrival of attributes in nx graphs

        graphsPB_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in list_graphs_nx]

        all_possible_columns_name: list[str] = ExtendedBoostingMatrix.__get_all_possible_node_attributes_in_the_dataset(
            list_graphs_nx,
            selected_paths)
        all_possible_columns_name.append("target")

        all_possible_attributes_from_single_graph: set[str] = ExtendedBoostingMatrix.__get_all_possible_attributes(
            list_graphs_nx)

        # initialize a list of sets that we will use to create the panda's dataframe
        list_rows: list[dict] = []

        for i, graph in enumerate(graphsPB_list):
            # dictionary that contains all the possible values for the same attribute in one graph
            accumulated_attributes = defaultdict(lambda: [])

            for labelled_path in selected_paths:
                numbered_paths_found_in_graph = graph.find_labelled_path(labelled_path=labelled_path)
                node_attributes: dict = {}

                for numbered_path in numbered_paths_found_in_graph:
                    node_attributes = ExtendedBoostingMatrix.__get_node_attributes_of_nx_graph(graph=list_graphs_nx[i],
                                                                                               node_id=numbered_path[
                                                                                                   -1])
                    # add the column counting the number of time labelled path is present in the graph
                    node_attributes["n_times_present"] = len(numbered_paths_found_in_graph)

                    for attr in node_attributes:
                        if attr in all_possible_attributes_from_single_graph:
                            # Accumulate the attribute values in a list
                            accumulated_attributes[
                                ExtendedBoostingMatrix.__get_column_name(labelled_path, attr)].append(
                                node_attributes[attr])

                # --------------------------------
            # Calculate the average of all the accumulated values
            complete_attributes = {attr: np.mean(values) if values else None for attr, values in
                                   accumulated_attributes.items()}

            # add response column
            complete_attributes["target"] = list_graphs_nx[i].graph[Settings.graph_label_variable]

            list_rows.append(complete_attributes)
        extended_boosting_matrix_df: pd.DataFrame = pd.DataFrame(list_rows)

        # some columns might be selected by the previous run in pattern boosting but not found in the new dataset
        missed_columns = list(set(all_possible_columns_name) - set(extended_boosting_matrix_df.columns))
        add_dataset = pd.DataFrame(np.nan, index=np.arange(extended_boosting_matrix_df.shape[0]),
                                   columns=missed_columns)
        # extended_boosting_matrix_df[missed_columns] = [np.nan]*len(missed_columns)
        extended_boosting_matrix_df = pd.concat([extended_boosting_matrix_df, add_dataset], axis=1)

        # convert into a sparse dataset
        if convert_to_sparse is True:
            extended_boosting_matrix_df = extended_boosting_matrix_df.astype(pd.SparseDtype(float, fill_value=np.nan))
        extended_boosting_matrix_df = ExtendedBoostingMatrix.sort_df_columns(extended_boosting_matrix_df)
        return extended_boosting_matrix_df

    @staticmethod
    def zero_all_elements_except_the_ones_referring_to_path(x_df: pd.DataFrame, y: pd.Series, path: tuple[int],
                                                            dict_of_interaction_constraints: dict) -> (
            pd.DataFrame, pd.Series):
        # it returns a pd.Dataframe that is a deepcopy of the one given in input, but with it puts nan values in the columns that are not referring to the input path
        assert (len(x_df) == len(y))
        assert isinstance(x_df, pd.DataFrame)
        assert isinstance(y, pd.Series)

        columns_to_keep = dict_of_interaction_constraints[path]

        list_of_paths_involved = [ExtendedBoostingMatrix.__parse_tuple_from_colname(column) for column in
                                  columns_to_keep]

        # Find the length of the longest tuples
        max_path_length = max(len(tup) for tup in list_of_paths_involved)

        # Find the indices of all tuples that have the maximum length
        indices_of_longest_tuples = [index for index, tup in enumerate(list_of_paths_involved) if
                                     len(tup) == max_path_length]

        columns_relative_only_to_last_path = [columns_to_keep[index] for index in indices_of_longest_tuples]

        nan_df = pd.DataFrame(np.nan, index=x_df.index, columns=x_df.columns)
        nan_df[columns_to_keep] = x_df[columns_to_keep]
        # nan_df = x_df.mask([column not in columns_to_keep for column in x_df.columns] & (x_df.notnull()), np.nan, inplace=False)

        # remove all the observations that have nan in the

        nan_df = pd.concat([nan_df, y], axis=1)

        nan_df.dropna(subset=columns_relative_only_to_last_path, inplace=True)

        zeroed_y = nan_df[y.name]
        zeroed_x_df = nan_df.drop(y.name, axis=1)

        return zeroed_x_df, zeroed_y
