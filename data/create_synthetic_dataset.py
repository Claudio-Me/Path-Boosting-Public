from data import data_reader
from classes.dataset import Dataset
import numpy as np


class CreateSyntheticDataset:
    '''
    It takes the given dataset and generates new labels from the formula y=c0p0 + c1p1 + c2p2 + ...
    Where p0...pn are the number of times target_path[0]...target_path[n] are present in the selected graph
    '''

    def __init__(self):
        self.target_paths = [
            (28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6), (28, 7, 6, 6), (28, 7, 6)

        ] # total length 21
        self.coef=[
            9,8,7,6,  # length 2
            4,5,6,  # length 3
            2, 3,  # length 4
            2,  # length 5
            3, 3, 3, 3, 3, 3,
            # length 6 with previous
            2,  # tuple that can be repeated
            2,
            2, 2, 2
        ]
        not_used = [(47, 7), (42, 7, 6), (75, 8), (29,), (75, 17), (78, 7, 6), (47,), (79, 17), (40,), (28, 7),
                    (28, 16), (46, 6), (23, 7, 6), (28, 7, 7), (79, 6, 6), (75, 7, 6), (75, 7), (75, 16),
                    (79, 7), (46, 6, 7), (79, 16), (23, 8), (42, 6), (42, 15), (45, 6, 7), (42, 8), (28, 7, 6),
                    (30, 8), (30, 17), (75, 6), (42, 7, 7), (79, 6), (79, 15), (23, 7), (78, 16),
                    (28, 6), (28, 15), (28, 16, 6), (22, 6, 6), (44, 15, 6), (28, 8), (42, 7), (28, 17),
                    (28, 35), (30, 7), (79, 7, 6), (30, 16), (75, 15), (46, 6, 6), (75,), (77,), (27,), (22,), (30,),
                    (23,), (24,), (79,), (74,), (28,), (46,), (73,), (45,), (48,),
                    (42,), (26,), (44,), (25,), (78,), (80,)]
        self.variance = 1

    def crate_datsaset_from_5k_selection_graph(self):
        dataset = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs")
        number_paths_counting = np.array(
            [[graph.number_of_time_path_is_present_in_graph(path) for path in self.target_paths] for
             graph in dataset.graphs_list])

        new_labels = self.__formula_new_labels(number_paths_counting)
        dataset.labels = list(new_labels)

        a = number_paths_counting.sum(axis=1)
        z = np.count_nonzero(a)
        new_graphs_list = []
        new_labels_list = []

        for i in range(len(a)):
            if a[i] != 0:
                new_graphs_list.append(dataset.graphs_list[i])
                new_labels_list.append(new_labels[i])

        return Dataset(graphs_list=new_graphs_list, labels=new_labels_list)

    def __formula_new_labels(self, number_paths_counting):
        coefficients = np.random.uniform(10, 20, len(self.target_paths))
        assert len(coefficients) == len(self.target_paths)

        # ------------------------------------------------------------------------------------------
        # can be rewritten using vector multiplication, but I don't have internet now to check how to do it
        # also if number_of_paths_counting is a matrix we can use matrix multiplications
        number_paths_counting = np.array(number_paths_counting)
        y = np.matmul(number_paths_counting, coefficients)

        # add random noise

        noise = np.random.normal(0, self.variance, len(y))
        y = y + noise

        # ------------------------------------------------------------------------------------------
        return y
