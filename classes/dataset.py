from classes.graph import GraphPB
from settings import Settings
from sklearn.model_selection import train_test_split
import networkx as nx
import numbers
import warnings


class Dataset:

    def __init__(self, graphs_list: list, labels: list = None):
        if isinstance(graphs_list[0], GraphPB):
            self.graphs_list = graphs_list
        elif isinstance(graphs_list[0], nx.classes.multigraph.MultiGraph):
            if labels is None:
                self.graphs_list = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in graphs_list]
            else:
                self.graphs_list = [GraphPB.from_GraphNX_to_GraphPB(graph, label) for graph, label in
                                    zip(graphs_list, labels)]

        else:
            raise TypeError("Graph format not recognized")

        if labels is None:
            self.labels: list = [graph.label_value for graph in self.graphs_list]
        else:
            self.labels: list = labels
        if not (isinstance(self.labels[0], numbers.Number)):
            warnings.warn("Warning, labels of the graphs are not numbers")

    def get_first_n_entries(self, n):
        return Dataset(self.graphs_list[:n], self.labels[:n])

    def split_dataset(self, test_size):
        x_train, x_test, y_train, y_test = train_test_split(self.graphs_list, self.labels, test_size=test_size)
        train_dataset = Dataset(x_train, y_train)
        test_dataset = Dataset(x_test, y_test)
        return train_dataset, test_dataset

    def merge_datasets(self, new_dataset):
        self.graphs_list = self.graphs_list + new_dataset.graphs_list
        self.labels = self.labels + new_dataset.labels
