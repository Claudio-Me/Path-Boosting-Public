import numpy as np
import pandas as pd
from PyAstronomy import pyasl
from typing import List, Tuple
from classes.boosting_matrix import BoostingMatrix
import networkx as nx
from data import data_reader
from classes.graph import GraphPB
from collections import defaultdict
import matplotlib.pyplot as plt
import ast
from settings import Settings


class ExtendedBoostingMatrix:

    def __init__(self):
        self.df: pd.DataFrame | None = None

    def create_extend_boosting_matrix(self, selected_paths: list[tuple],
                                      dataset: list[nx.classes.multigraph.MultiGraph]):
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)

        # function to help the retrival of attributes in nx graphs

        self.graphs_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in dataset]

        if self.df is None:
            # this is useless
            columns_name: list[str] = self.__get_all_possible_node_attributes_in_the_dataset(dataset, selected_paths)
            columns_name.append("response_variable")
            self.df = pd.DataFrame(columns=columns_name)
        else:
            # concatenate two datasets
            raise Exception("Concat of two dataset is not implemented yet")

        all_possible_attributes_from_single_graph: set[str] = self.__get_all_possible_attributes(dataset)

        # initialize a list of sets that we will use to create the panda's dataframe
        list_rows: list[dict] = []

        for i, graph in enumerate(self.graphs_list):
            # dictionary that contains all the possible values for the same attribute in one graph
            accumulated_attributes = defaultdict(lambda: [])

            for labelled_path in selected_paths:
                numbered_paths_found_in_graph = graph.find_labelled_path(labelled_path=labelled_path)
                node_attributes: dict = {}

                for numbered_path in numbered_paths_found_in_graph:
                    node_attributes = self.__get_node_attributes_of_nx_graph(graph=dataset[i],
                                                                             node_id=numbered_path[-1])

                    for attr in node_attributes:
                        if attr in all_possible_attributes_from_single_graph:
                            # Accumulate the attribute values in a list
                            accumulated_attributes[self.__get_column_name(labelled_path, attr)].append(
                                node_attributes[attr])

                # --------------------------------
            # Calculate the average of all the accumulated values
            complete_attributes = {attr: np.mean(values) if values else None for attr, values in
                                   accumulated_attributes.items()}

            # add response column
            complete_attributes["response_variable"] = dataset[i].graph[Settings.graph_label_variable]

            list_rows.append(complete_attributes)
        self.df = pd.DataFrame(list_rows)

        # convert into a sparse dataset
        self.df = self.df.astype(pd.SparseDtype(float, fill_value=np.nan))
        self.sort_df_columns()

    @staticmethod
    def find_labels_in_nx_graph(graph: nx.Graph, path: List[int]):
        if len(path) == 1:
            return graph.get_label(path[0])

    @staticmethod
    def __get_all_possible_attributes(dataset: list[nx.classes.multigraph.MultiGraph]) -> set[str]:
        unique_attributes = set()
        for graph in dataset:
            for node, attributes in graph.nodes(data=True):
                unique_attributes.update(attributes.keys())
        if 'node_position' in unique_attributes:
            unique_attributes.remove('node_position')
        if 'node_label' in unique_attributes:
            unique_attributes.remove('node_label')

        return unique_attributes

    @staticmethod
    # finds all the possible attributes of the node
    def __get_node_attributes_of_nx_graph(graph, node_id) -> dict | None:
        if str(node_id) in graph.nodes:
            return graph.nodes[str(node_id)]
        else:
            return None

    @staticmethod
    def __get_all_possible_node_attributes_in_the_dataset(dataset: list[nx.classes.multigraph.MultiGraph],
                                                          selected_paths: list[tuple]) -> list[str]:
        unique_attributes = ExtendedBoostingMatrix.__get_all_possible_attributes(dataset)

        columns_name = set()
        for path in selected_paths:
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
        string_tuple, _ = column_name.split('_', 1)
        # Convert the string-tuple to an actual tuple
        tuple_of_ints = ast.literal_eval(string_tuple)
        # Create a sorting key: (length of the tuple, the tuple itself)
        return (len(tuple_of_ints), tuple_of_ints)

    def sort_df_columns(self):
        sorted_columns = sorted(self.df.columns, key=self.__column_sort_key)

        self.df = self.df.reindex(columns=sorted_columns)

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
