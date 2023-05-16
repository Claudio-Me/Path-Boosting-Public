import numpy as np

from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from settings import Settings
from data.synthetic_dataset import SyntheticDataset
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
from classes.analysis import Analysis
from data.load_dataset import load_dataset
from classes.dataset import Dataset
import sys


def different_rows(matrix):
    # Create a set to store the rows that have already been seen
    seen_rows = set()

    # Initialize a counter for the number of repeated rows
    repeated_rows = 0

    # Iterate through the rows of the matrix
    for row in matrix:
        # If the row has already been seen, increment the counter
        if tuple(row) in seen_rows:
            repeated_rows += 1
        # Otherwise, add the row to the set of seen rows
        else:
            seen_rows.add(tuple(row))

    return len(seen_rows)


def count_repeated_rows(matrix):
    # Create a set to store the rows that have already been seen
    seen_rows = set()

    # Initialize a counter for the number of repeated rows
    repeated_rows = 0

    # Iterate through the rows of the matrix
    for row in matrix:
        # If the row has already been seen, increment the counter
        if tuple(row) in seen_rows:
            repeated_rows += 1
        # Otherwise, add the row to the set of seen rows
        else:
            seen_rows.add(tuple(row))

    return repeated_rows


def append_matrix_rows(matrix1, matrix2):
    # Check that the matrices have the same number of columns
    return np.vstack((matrix1, matrix2))


if __name__ == '__main__':
    # Testing()
    print("Dataset name: ", Settings.dataset_name)
    dataset = load_dataset()







    for tmp in range(1):
        print("training on metal center number: ", Settings.considered_metal_center[0])
        dataset = load_dataset()

      #----------------------------------------------------------------------------------------------------------
      #print("size of dataset: ", asizeof.asizeof(dataset))
      #-----------------------------------------------------------------------------------------------------------
        new_dataset_list=[]
        print("selecting only graphs with specific metal center")

        for graph in dataset.get_graphs_list():
          #46 is the most common metal center

          if graph.node_to_label[graph.metal_center[0]]==Settings.considered_metal_center[0]:
               if len(graph.adj_list[graph.metal_center[0]]) != 0:
                    new_dataset_list.append(graph)
        del dataset
        if len(new_dataset_list)==0:
            sys.exit("No metal centers found")
        dataset=Dataset(new_dataset_list)


    # ----------------------------------------------------------------------------------------------------------
        print("len of dataset",len(new_dataset_list))
        #print("size of dataset: ", asizeof.asizeof(dataset))
    # -----------------------------------------------------------------------------------------------------------




    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                      random_split_seed=Settings.random_split)


    pattern_boosting = PatternBoosting()
    # test_dataset.labels=np.zeros(len(test_dataset.labels))
    pattern_boosting.training(train_dataset, test_dataset)

    data_reader.save_data(pattern_boosting, filename="pattern_boosting", directory="results")

    analysis = Analysis()
    analysis.load_and_analyze(directory=data_reader.get_save_location(folder_relative_path="results"),
                              show=Settings.show_analysis,
                              save=Settings.save_analysis)
    # analysis.all_analysis(pattern_boosting=pattern_boosting, synthetic_dataset=synthetic_dataset, show=False, save=True)

    # ------------------------------------------------------------------------------------------------------------------

    # additional analysis
    # check number of repeated rows
    train_boosting_matrix = pattern_boosting.create_boosting_matrix_for(train_dataset)
    test_boosting_matrix = pattern_boosting.create_boosting_matrix_for(test_dataset)

    print("--------------------------------------------------------------------------------")
    print("Repeated rows in final training boosting matrix")
    print(count_repeated_rows(train_boosting_matrix))
    print("Different rows in train matrix: ", different_rows(train_boosting_matrix))

    print("Repeated rows in final test boosting matrix")
    print(count_repeated_rows(test_boosting_matrix))

    print("Different rows in test matrix: ", different_rows(test_boosting_matrix))

    # ------------------------------------------------------------------------------------------------------------------

    '''
    if Settings.estimation_type is EstimationType.classification:
        dataset = data_reader.read_dataset_and_labels_from_csv("data/5k-selection-graphs", "tmQMg_5k_bin_class.csv")

    elif Settings.estimation_type is EstimationType.regression:
        dataset = data_reader.read_data_from_directory("data/5k-selection-graphs")

    else:
        raise TypeError("Wrong estimation type")

    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)
    pattern_boosting = PatternBoosting()
    pattern_boosting.training(train_dataset, test_dataset)
    '''
