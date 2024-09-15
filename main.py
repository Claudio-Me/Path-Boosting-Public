import functools
import logging
import sys
import tracemalloc
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool

from classes.analysis_patternboosting import AnalysisPatternBoosting
from classes.analysis_wrapper_pattern_boosting import \
    AnalysisWrapperPatternBoosting
from classes.dataset import Dataset
# from pympler import asizeof
from classes.graph import GraphPB
from classes.pattern_boosting import PatternBoosting
from classes.testing.testing import Testing
from classes.wrapper_pattern_boosting import WrapperPatternBoosting
from data import data_reader
from data.load_dataset import load_dataset
from data.synthetic_dataset import SyntheticDataset
from settings import Settings
import os

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/log_{now}.log"




    print("Number of CPU's: ", Settings.max_number_of_cores)
    print("Dataset name: ", Settings.dataset_name)

    dataset = load_dataset()

    # ----------------------------------------------------------------------------------------------------------
    # TODO remove the following line, we are just duplicating the dataset
    # TODO remove memory tracer

    if Settings.plot_log_info is True:
        # Ensure that the log directory exists and initalize logger
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.basicConfig(filename=filename, level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Testing()

        logger.info("Starting the program")

        # starting the memory monitoring
        tracemalloc.start()
        print(tracemalloc.get_traced_memory())

        Settings.log_principal_settings_values(logger=logger)



    # ----------------------------------------------------------------------------------------------------------



    train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                      random_split_seed=Settings.random_split_test_dataset_seed)

    # wrapper pattern boosting:
    if Settings.wrapper_boosting is True:
        wrapper_pattern_boosting = WrapperPatternBoosting()
        wrapper_pattern_boosting.train(train_dataset, test_dataset)

        final_test_error = wrapper_pattern_boosting.get_wrapper_test_error()
        print("len final test error", len(final_test_error))
        final_test_error = final_test_error[-1]

    else:
        # pattern boosting
        pattern_boosting = PatternBoosting()
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
        string = str(Settings.considered_metal_centers) + '-'
        string += str(final_test_error) + '\n'
        print(string)
        sys.stdout = original_stdout  # Reset the standard output to its original value

    if Settings.wrapper_boosting is True:
        print("Number of tained models: ", len(wrapper_pattern_boosting.get_trained_pattern_boosting_models()))
        data_reader.save_data(wrapper_pattern_boosting, filename="wrapper_pattern_boosting", directory="results")
    else:
        data_reader.save_data(pattern_boosting, filename="pattern_boosting", directory="results")

    if Settings.wrapper_boosting is True:
        if Settings.dataset_name == "5k_synthetic_dataset":
            synthetic_dataset = SyntheticDataset()
        else:
            synthetic_dataset = None
        if Settings.save_analysis or Settings.show_analysis:
            analysis = AnalysisWrapperPatternBoosting(wrapper_pattern_boosting, save=Settings.save_analysis,
                                                      show=Settings.show_analysis)
            analysis.plot_all_analysis(n=Settings.n_of_paths_importance_plotted, synthetic_dataset=synthetic_dataset)


    else:
        if Settings.save_analysis or Settings.show_analysis:
            analysis = AnalysisPatternBoosting()
            analysis.load_and_analyze(directory=data_reader.get_save_location(folder_relative_path="results",
                                                                              unique_subfolder=True),
                                      show=Settings.show_analysis,
                                      save=Settings.save_analysis)

        '''
        if Settings.dataset_name == "5k_synthetic_dataset":
            analysis.all_analysis(pattern_boosting=pattern_boosting, synthetic_dataset=synthetic_dataset, show=False,
                                  save=True)
        '''
