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
from data.load_dataset import split_dataset_by_metal_centers
from classes.dataset import Dataset
# from pympler import asizeof
from classes.graph import GraphPB
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

    # ----------------------------------------------------------------------------------------------------------
    # print("size of dataset: ", asizeof.asizeof(dataset))
    # -----------------------------------------------------------------------------------------------------------

    print("selecting only graphs with specific metal center")

    splitted_datasets_list = split_dataset_by_metal_centers(dataset)


    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                      random_split_seed=Settings.random_split_test_dataset_seed)
    # paralelize after this
    pattern_boosting = PatternBoosting()
    # test_dataset.labels=np.zeros(len(test_dataset.labels))
    pattern_boosting.training(train_dataset, test_dataset)
    final_test_error = pattern_boosting.test_error[-1]

    print("final test error:\n", final_test_error)

    saving_location = data_reader.get_save_location(file_name="final_test_error", file_extension=".txt",
                                                    folder_relative_path='results')

    original_stdout = sys.stdout
    with open(saving_location, 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        string = str(Settings.considered_metal_centers[0]) + '-'
        string += str(final_test_error) + '\n'
        print(string)
        sys.stdout = original_stdout  # Reset the standard output to its original value

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
