import numpy as np

from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from data.create_synthetic_dataset import CreateSyntheticDataset
from classes.analysis import Analysis

if __name__ == '__main__':
    Testing()

    #dataset = data_reader.load_dataset_from_binary(filename="5k_synthetic_dataset")
    dataset = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs")
    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size)


    pattern_boosting = PatternBoosting()
    # test_dataset.labels=np.zeros(len(test_dataset.labels))
    pattern_boosting.training(train_dataset, test_dataset)


    #------------------------------------------------------------------------------------------------------------------
    # debugging

    # check number of repeated rows
    train_boosting_matrix=pattern_boosting.create_boosting_matrix_for(train_dataset)
    test_boosting_matrix=pattern_boosting.create_boosting_matrix_for(test_dataset)


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
    print("--------------------------------------------------------------------------------")
    print("Repeated rows in final training boosting matrix")
    print(count_repeated_rows(train_boosting_matrix))
    print("Repeated rows in final test boosting matrix")
    print(count_repeated_rows(test_boosting_matrix))

    def append_matrix_rows(matrix1, matrix2):
        # Check that the matrices have the same number of columns
        return np.vstack((matrix1, matrix2))
    big_matrix= append_matrix_rows(train_boosting_matrix,test_boosting_matrix)

    print("Repeated rows in big boosting matrix")
    print(count_repeated_rows(big_matrix))




    selected_paths=[]
    for selected_column in pattern_boosting.boosting_matrix.already_selected_columns:
        selected_paths.append(pattern_boosting.boosting_matrix.header[selected_column])
    synth=CreateSyntheticDataset()
    # count the number of really important paths that have been selected by the algorithm
    counter=0
    selected_paths=set(selected_paths)
    for target_path in synth.target_paths:
        if target_path in selected_paths:
            counter +=1
    print("Number of target paths that have been spotted")
    print(counter)

    print("Total number of target paths: ", len(synth.target_paths))

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
