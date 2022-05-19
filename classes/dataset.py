from classes.graph import GraphPB
from settings import Settings
import networkx as nx
import numbers
import warnings


class Dataset:

    def __init__(self, graphs_list: list):
        if isinstance(graphs_list[0], GraphPB):
            self.graphs_list = graphs_list
        elif isinstance(graphs_list[0], nx.classes.multigraph.MultiGraph):
            self.graphs_list = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in graphs_list]
        else:
            raise TypeError("Graph format not recognized")

        self.labels: list = [graph.label_value for graph in self.graphs_list]
        if not (isinstance(self.labels[0], numbers.Number)):
            warnings.warn("Warning, labels of the graphs are not numbers")
