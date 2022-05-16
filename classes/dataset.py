from graph import GraphPB
import networkx as nx

class Dataset:

    def __init__(self, graphs_list):
        # dataset is assumed to be a list of graphs
        if isinstance(graphs_list[0], GraphPB):
            self.graphs_list = graphs_list
        elif isinstance(graphs_list[0], nx.classes.multigraph.MultiGraph):
            self.graphs_list = [GraphPB.from_GraphNX_to_GraphPB(graph) for graph in graphs_list]
        else:
            raise TypeError("Graph format not recognized")

        response_variable=
