import sys

sys.path.insert(0, "../")
from classes.testing.testing import Testing
from classes.pattern_boosting import PatternBoosting
from data import data_reader
from classes.analysis_wrapper_pattern_boosting import AnalysisWrapperPatternBoosting
from settings import Settings
from data.synthetic_dataset import SyntheticDataset
from classes.analysis_patternboosting import AnalysisPatternBoosting
from data.load_dataset import load_dataset
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
import sys

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

if __name__ == '__main__':
    # Testing()
    settings = Settings()

    print("Number of CPU's: ", settings.max_number_of_cores)
    print("Dataset name: ", settings.dataset_name)

    dataset = load_dataset()

    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, settings.test_size,
                                                                      random_split_seed=settings.random_split_test_dataset_seed)

    # wrapper pattern boosting:
    if settings.wrapper_boosting is True:
        wrapper_pattern_boosting = WrapperPatternBoosting(metal_center_list=settings.considered_metal_centers,
                                                          settings=settings)
        wrapper_pattern_boosting.train(train_dataset, test_dataset)

        final_test_error = wrapper_pattern_boosting.get_wrapper_test_error()
        print("len final test error", len(final_test_error))
        final_test_error = final_test_error[-1]

    else:
        # pattern boosting
        pattern_boosting = PatternBoosting(settings=settings)
        pattern_boosting.training(train_dataset, test_dataset)

        final_test_error = pattern_boosting.test_error[-1]

    print("final test error:\n", final_test_error)

    saving_location = data_reader.get_save_location(file_name="final_test_error", file_extension=".txt",
                                                    folder_relative_path='results', unique_subfolder=True)

    print("Saving location:")
    print(saving_location)

    original_stdout = sys.stdout
    with open(saving_location, 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        string = str(settings.considered_metal_centers) + '-'
        string += str(final_test_error) + '\n'
        print(string)
        sys.stdout = original_stdout  # Reset the standard output to its original value

    if settings.wrapper_boosting is True:
        print("Number of tained models: ", len(wrapper_pattern_boosting.get_trained_pattern_boosting_models()))
        data_reader.save_data(wrapper_pattern_boosting, filename="wrapper_pattern_boosting", directory="results")
    else:
        data_reader.save_data(pattern_boosting, filename="pattern_boosting", directory="results")

    if settings.wrapper_boosting is True:
        if settings.dataset_name == "5k_synthetic_dataset":
            synthetic_dataset = SyntheticDataset(settings=settings)
        else:
            synthetic_dataset = None
        analysis = AnalysisWrapperPatternBoosting(wrapper_pattern_boosting=wrapper_pattern_boosting, settings=settings,
                                                  save=settings.save_analysis, show=settings.show_analysis)
        analysis.plot_all_analysis(n=settings.n_of_paths_importance_plotted, synthetic_dataset=synthetic_dataset)


    else:
        analysis = AnalysisPatternBoosting(pattern_boosting=pattern_boosting, settings=settings)
        analysis.load_and_analyze(directory=data_reader.get_save_location(folder_relative_path="results",
                                                                          unique_subfolder=True),
                                  show=settings.show_analysis,
                                  save=settings.save_analysis)

        '''
        if settings.dataset_name == "5k_synthetic_dataset":
            analysis.all_analysis(pattern_boosting=pattern_boosting, synthetic_dataset=synthetic_dataset, show=False,
                                  save=True)
        '''
