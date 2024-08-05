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
    def __int__(self):
        self.df: pd.DataFrame | None = None

    def create_extend_boosting_matrix(self, selected_paths: list, dataset: list[nx.classes.multigraph.MultiGraph]):
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        assert isinstance(selected_paths, list)
        new_columns: pd.DataFrame | None = None

        # function to help the retrival of attributes in nx graphs

        self.graphs_list: list[GraphPB] = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in dataset]
        for i, graph in enumerate(self.graphs_list):
            if new_columns is None:
                all_possible_attributes: list = self.__get_all_possible_attributes(dataset[i])
                # remove node_position from the list
                if 'node_position' in all_possible_attributes:
                    all_possible_attributes.remove('node_position')
                    all_possible_attributes.remove('node_label')

                new_columns = pd.DataFrame(columns=all_possible_attributes)

            for labelled_path in selected_paths:
                paths = graph.find_labelled_path(labelled_path=labelled_path)
                node_attributes: dict = {}

                accumulated_attributes = defaultdict(lambda: [])

                for path in paths:
                    node_attributes = self.__get_node_attributes_of_nx_graph(graph=dataset[i], node_id=path[-1])

                    for attr in all_possible_attributes:
                        # Skip if the attribute is not present for this node
                        if attr not in node_attributes:
                            continue

                        # Accumulate the attribute values in a list
                        accumulated_attributes[attr].append(node_attributes[attr])

                # --------------------------------
            # Calculate the average of all the accumulated values
            complete_attributes = {attr: np.mean(values) if values else None for attr, values in
                                   accumulated_attributes.items()}

            new_columns = pd.concat([new_columns, pd.DataFrame([complete_attributes])], ignore_index=True)
        self.df = new_columns

    @staticmethod
    def find_labels_in_nx_graph(graph: nx.Graph, path: List[int]):
        if len(path) == 1:
            return graph.get_label(path[0])

    @staticmethod
    def __get_all_possible_attributes(graph) -> list:
        unique_attributes = set()
        for node, attributes in graph.nodes(data=True):
            unique_attributes.update(attributes.keys())
        return list(unique_attributes)

    @staticmethod
    # finds all the possible attributes of the node
    def __get_node_attributes_of_nx_graph(graph, node_id) -> dict | None:
        if str(node_id) in graph.nodes:
            return graph.nodes[str(node_id)]
        else:
            return None
