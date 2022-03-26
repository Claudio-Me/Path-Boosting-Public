import networkx as nx


def read_data(dataset_name):
    directory = "data/"
    return nx.read_gml(directory + dataset_name)


