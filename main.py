import numpy as np

from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from settings import Settings
from data.synthetic_dataset import SyntheticDataset
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
from classes.analysis_patternboosting import AnalysisPatternBoosting
from data.load_dataset import load_dataset
from data.load_dataset import split_dataset_by_metal_centers
from classes.dataset import Dataset
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
# from pympler import asizeof
from classes.graph import GraphPB
import sys
from multiprocessing.dummy import Pool as ThreadPool
import functools



if __name__ == '__main__':
    # Testing()
    print("Dataset name: ", Settings.dataset_name)

    dataset = load_dataset()

    # ----------------------------------------------------------------------------------------------------------
    # print("size of dataset: ", asizeof.asizeof(dataset))
    # -----------------------------------------------------------------------------------------------------------


    """
    splitted_train_datasets_list = split_dataset_by_metal_centers(train_dataset)

    splitted_test_datasets_list = split_dataset_by_metal_centers(test_dataset)

    """
    # paralelize after this
    # Parallelization template
    # ------------------------------------------------------------------------------------------------------------
    """
    pool = ThreadPool(min(10, len(Settings.considered_metal_centers)))

    array_of_outputs = pool.map(
        functools.partial(function_to_parallelize,
                          param_1=param_1,
                          param_2=param_2),
        list_over_wich_we_parallelize)

    xgb_test_err, xgb_train_err, variable_importance = zip(*xgb_err)
    """
    # ------------------------------------------------------------------------------------------------------------

    # part to be parellized
    #-------------------------------------------------------------------------------------------------
    """
    pattern_boosting = PatternBoosting()
    # test_dataset.labels=np.zeros(len(test_dataset.labels))
    pattern_boosting.training(train_dataset, test_dataset)
    final_test_error = pattern_boosting.test_error[-1]
    """
    #-------------------------------------------------------------------------------------------------

    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                      random_split_seed=Settings.random_split_test_dataset_seed)

    wrapper_pattern_boosting=WrapperPatternBoosting()
    wrapper_pattern_boosting.train(train_dataset,test_dataset)

    final_test_error=wrapper_pattern_boosting.get_wrapper_test_error()[-1]


    pattern_boosting = PatternBoosting()
    pattern_boosting.training(train_dataset, test_dataset)
    final_test_error = pattern_boosting.test_error[-1]


    print("final test error:\n", final_test_error)

    saving_location = data_reader.get_save_location(file_name="final_test_error", file_extension=".txt",
                                                    folder_relative_path='results')

    print("Saving location:")
    print(saving_location)

    original_stdout = sys.stdout
    with open(saving_location, 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        string = str(Settings.considered_metal_centers) + '-'
        string += str(final_test_error) + '\n'
        print(string)
        sys.stdout = original_stdout  # Reset the standard output to its original value


    data_reader.save_data(wrapper_pattern_boosting, filename="wrapper_pattern_boosting", directory="results")
    #data_reader.save_data(pattern_boosting, filename="pattern_boosting", directory="results")



    """
    analysis = AnalysisPatternBoosting()
    analysis.load_and_analyze(directory=data_reader.get_save_location(folder_relative_path="results"),
                              show=Settings.show_analysis,
                              save=Settings.save_analysis)
    """
    # analysis.all_analysis(pattern_boosting=pattern_boosting, synthetic_dataset=synthetic_dataset, show=False, save=True)

