from testing.testing_from_array_label_to_dictionary import testing
import networkx as nx


def read_data(dataset_name):
    directory = "data/"
    return nx.read_gml(directory + dataset_name)


if __name__ == '__main__':
    my_graph = read_data("LALMER.gml")
    u = my_graph.edge_attr_dict_factory
    my_graph.sticazzi=3
    print(u)
