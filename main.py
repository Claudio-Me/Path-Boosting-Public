import numpy as np

from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from settings import Settings
from data.synthetic_dataset import SyntheticDataset
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
from classes.analysis import Analysis


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

    dataset = data_reader.load_dataset()

    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)

    save_split_test_and_train=False
    load_split_test_and_train=False

    # save
    if save_split_test_and_train is True:
        data_reader.save_dataset_in_binary_file(train_dataset, filename=Settings.dataset_name+"_train")
        data_reader.save_dataset_in_binary_file(test_dataset, filename=Settings.dataset_name+"_test")

    # load
    if load_split_test_and_train is True:
        train_dataset = data_reader.load_dataset_from_binary(filename=Settings.dataset_name + "_train")
        test_dataset = data_reader.load_dataset_from_binary(filename=Settings.dataset_name + "_train")

    pattern_boosting = PatternBoosting()
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

    big_matrix = append_matrix_rows(train_boosting_matrix, test_boosting_matrix)

    print("Repeated rows in big boosting matrix")
    print(count_repeated_rows(big_matrix))

    # ------------------------------------------------------------------------------------------------------------------


    print("End")


