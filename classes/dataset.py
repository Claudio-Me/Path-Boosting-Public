import numpy as np
from lightgbm import Dataset

from classes.graph import GraphPB
from settings import Settings
from sklearn.model_selection import train_test_split
import networkx as nx
import numbers
import warnings


class Dataset:

    def __init__(self, graphs_list: list | None, labels: list = None):
        # in case graph list is None the object is used only by a specific method in wrapper_pattern_boosting
        if graphs_list is None:
            self.graphs_list = []
        else:
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
                self.labels: list = [graph.label for graph in self.graphs_list]
            else:
                self.labels: list = labels
            if not (isinstance(self.labels[0], numbers.Number)):
                warnings.warn("Warning, labels of the graphs are not numbers")

    def remove(self, graph: GraphPB)->bool:
        try:
            index = self.graphs_list.index(graph)
        except:
            return False
        del self.graphs_list[index]
        del self.labels[index]
        return True


    def add(self, graph: GraphPB):
        self.graphs_list.append(graph)
        self.labels.append(graph.label)

    def get_first_n_entries(self, n):
        return Dataset(self.graphs_list[:n], self.labels[:n])

    def split_dataset(self, test_size, random_split_seed=None):
        if test_size == 0:
            return self, None
        x_train, x_test, y_train, y_test = train_test_split(self.graphs_list, self.labels, test_size=test_size,
                                                            random_state=random_split_seed)
        train_dataset = Dataset(x_train, y_train)
        test_dataset = Dataset(x_test, y_test)

        return train_dataset, test_dataset

    def merge_datasets(self, new_dataset):
        self.graphs_list = self.graphs_list + new_dataset.graphs_list
        self.labels = self.labels + new_dataset.labels

    def get_graphs_list(self)->list[GraphPB]:
        return self.graphs_list



    def get_labels(self):
        return self.labels

    def get_dimension(self) -> int:
        # note: I use len of graph list instead of len labels because sometimes the dataset may have None as labels
        return len(self.graphs_list)

    def __str__(self):
        average_label = np.mean(self.labels)
        string = "Average value of label: " + str(average_label) + "\n"
        string = string + "Dataset dimension: " + str(self.get_dimension()) + "\n"
        return string

    def get_graph_number(self, number):
        return self.get_graphs_list()[number]



