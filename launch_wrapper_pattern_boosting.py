import numpy as np

from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from classes.analysis_wrapper_pattern_boosting import AnalysisWrapperPatternBoosting
from settings import Settings
from data.synthetic_dataset import SyntheticDataset
from classes.enumeration.estimation_type import EstimationType
from data.synthetic_dataset import SyntheticDataset
from classes.analysis_patternboosting import AnalysisPatternBoosting
from data.load_dataset import load_dataset
from classes.dataset import Dataset
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
# from pympler import asizeof
from classes.graph import GraphPB
import sys
from multiprocessing.dummy import Pool as ThreadPool
import functools

# Seggings
if True:
    Settings.maximum_number_of_steps = 30

    Settings.save_analysis = True
    Settings.show_analysis = False

    Settings.dataset_name = "60k_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    Settings.generate_new_dataset = False

    # in the error graph Print only the last N learners
    Settings.tail = 1000

    Settings.wrapper_boosting = True

    # used in wrapped boosting to specify the centers over which split the dataset
    if Settings.wrapper_boosting is True:
        Settings.considered_metal_centers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                                             39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                                             57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                                             72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                                             89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
                                             # actinides
                                             104, 105, 106, 107, 108, 109, 110, 111, 112]
    else:
        Settings.considered_metal_centers = None
        print("nfjskdfsajkfndsjkfndsjkgndsjgndsjknsdjif")

    # do not expand if the paths are longer than this amount
    Settings.max_path_length = 102

    # portion of the whole dataset that needs to be used as test dataset
    Settings.test_size = 0.01

    Settings.target_train_error = 0.0000001

    # it works only if "algorithm" is Xgb_step
    Settings.update_features_importance_by_comparison = False

# load wrapper pattern boosting
directory = data_reader.get_save_location(file_name="wrapper_pattern_boosting", file_extension=".pkl",
                                          folder_relative_path="results/wrapper_boosting_60k_dataset",
                                          unique_subfolder=False)
wrapper_pattern_boosting = data_reader.load_data(directory=directory)

wrapper_pattern_boosting.re_train()
error = wrapper_pattern_boosting.get_wrapper_test_error()
final_test_error = wrapper_pattern_boosting.get_wrapper_test_error()
print("len final test error", len(final_test_error))
final_test_error = final_test_error[-1]

print("final test error:\n", final_test_error)

saving_location = data_reader.get_save_location(file_name="final_test_error", file_extension=".txt",
                                                folder_relative_path='results/wrapper_boosting_60k_dataset',
                                                unique_subfolder=False)

print("Saving location:")
print(saving_location)

original_stdout = sys.stdout
with open(saving_location, 'a') as f:
    sys.stdout = f  # Change the standard output to the file we created.
    string = str(Settings.considered_metal_centers) + '-'
    string += str(final_test_error) + '\n'
    print(string)
    sys.stdout = original_stdout  # Reset the standard output to its original value

print("Number of trained models: ", len(wrapper_pattern_boosting.get_trained_pattern_boosting_models()))
data_reader.save_data(wrapper_pattern_boosting, filename="wrapper_pattern_boosting",
                      directory="results/wrapper_boosting_60k_dataset",
                      create_unique_subfolder=False)
