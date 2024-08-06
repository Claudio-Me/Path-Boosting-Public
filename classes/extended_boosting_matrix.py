import numpy as np
import pandas as pd
from PyAstronomy import pyasl
from typing import List, Tuple
from classes.boosting_matrix import BoostingMatrix
import networkx as nx
from data import data_reader
from classes.graph import GraphPB
from collections import defaultdict


class ExtendedBoostingMatrix:

    def __init__(self):
        self.df: pd.DataFrame | None = None


    def create_extend_boosting_matrix(self, selected_paths: list[tuple],
                                      dataset: list[nx.classes.multigraph.MultiGraph]):
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)

        # function to help the retrival of attributes in nx graphs

        columns_name: list[str] = self.__get_all_possible_node_attributes_in_the_dataset(dataset, selected_paths)

        self.graphs_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in dataset]

        if self.df is None:
            self.df = pd.DataFrame(columns=columns_name)
        else:
            # concatenate two datasets
            self.df = pd.concat([self.df, pd.DataFrame(columns=columns_name)]).drop_duplicates().reset_index(drop=True)

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
                            accumulated_attributes[self.__get_column_name(labelled_path,attr)].append(node_attributes[attr])

                # --------------------------------
            # Calculate the average of all the accumulated values
            complete_attributes = {attr: np.mean(values) if values else None for attr, values in
                                   accumulated_attributes.items()}

            list_rows.append(complete_attributes)
        self.df = pd.DataFrame(list_rows)

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
            for attribute in unique_attributes:
                columns_name.add(ExtendedBoostingMatrix.__get_column_name(path, attribute))

        return sorted(columns_name)

    @staticmethod
    def __get_column_name(path: tuple, attribute: str):
        return str(path) + '_' + attribute
