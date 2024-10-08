import sys
import os
from collections import Counter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

sys.path.insert(0, "../")
from jupiter_notebook_functions import *
from analysis_article.set_default_settings import set_default_settings


def cross_validation(number_of_simulations=200, k_folds=5, scenario=1, patience=3, dataset_name="5k_synthetic_dataset",
                     noise_variance=0.2, maximum_number_of_steps=None,
                     save_fig=False, use_wrapper_boosting=None, show_settings=True):
    set_default_settings()

    Settings.noise_variance = noise_variance

    Settings.scenario = scenario
    Settings.set_scenario(scenario)

    Settings.save_analysis = False
    Settings.show_analysis = False
    Settings.dataset_name = dataset_name  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    if Settings.dataset_name == "5k_synthetic_dataset":
        Settings.generate_new_dataset = True
        fig_name = "test_error_cross_validation_scenario_" + str(Settings.scenario)
    elif Settings.dataset_name == "5_k_selection_graphs" or Settings.dataset_name == "60k_dataset":
        Settings.generate_new_dataset = False
        fig_name = "test_error_cross_validation"
        Settings.wrapper_boosting = True

    Settings.wrapper_boosting = use_wrapper_boosting

    if (
            scenario == 1 or scenario == 2) and Settings.dataset_name == "5k_synthetic_dataset":
        Settings.wrapper_boosting = False
    elif scenario == 3 and Settings.dataset_name == "5k_synthetic_dataset":
        Settings.wrapper_boosting = True

    if maximum_number_of_steps is None:
        if scenario == 1:
            Settings.maximum_number_of_steps = 80
        elif scenario == 2:
            Settings.maximum_number_of_steps = 150
        elif scenario == 3:
            Settings.maximum_number_of_steps = 350
    else:
        Settings.maximum_number_of_steps = maximum_number_of_steps

    list_overfitting_iterations = []
    list_of_test_errors: list[list[float]] = []
    list_n_selected_paths = []
    list_oracle_test_error = []
    list_test_errors_cross_validation_list = []

    # launch cross validation
    for i in range(number_of_simulations):
        print("iteration number ", i)
        print(Settings.maximum_number_of_steps)

        dataset = load_dataset()
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                          random_split_seed=Settings.random_split_test_dataset_seed)
        if Settings.dataset_name == "5k_synthetic_dataset":
            synthetic_dataset = SyntheticDataset()
            oracle_test_error = synthetic_dataset.oracle_model_evaluate(
                graphs_list=test_dataset.get_graphs_list(),
                labels=test_dataset.get_labels())

            list_oracle_test_error.append(oracle_test_error)

        overfitting_iteration, test_error, n_selected_paths, test_errors_cross_validation_list = perform_cross_validation(
            train_dataset, test_dataset,
            k=k_folds,
            random_seed=Settings.cross_validation_k_fold_seed,
            patience=patience)

        list_overfitting_iterations.append(overfitting_iteration)
        list_of_test_errors.append(test_error)
        list_n_selected_paths.append(n_selected_paths)
        list_test_errors_cross_validation_list.append(test_errors_cross_validation_list)

    if show_settings is True:
        Settings.print_principal_values()
    data_reader.save_data(data=list_test_errors_cross_validation_list, filename='test_errors_cross_validation_list',
                          directory='results/cross_validation', create_unique_subfolder=True)

    saving_location = data_reader.get_save_location(file_name="overfitting_iteration", file_extension=".txt",
                                                    folder_relative_path='results/cross_validation',
                                                    unique_subfolder=True)
    print(saving_location)
    with open(saving_location, "a") as f:
        print("max number of steps")
        print(Settings.maximum_number_of_steps)

        print("average overfitting iteration:", file=f)
        print(np.average(list_overfitting_iterations), file=f)

        print("max overfitting iteration", file=f)
        print(max(list_overfitting_iterations), file=f)

        print("std overfitting iteration:", file=f)
        print(np.std(list_overfitting_iterations), file=f)

        print("average test error, standard error", file=f)
        final_test_error_list = []
        for array_test_error in list_of_test_errors:
            final_test_error_list.append(array_test_error[-1])

        print(np.average(final_test_error_list), np.std(final_test_error_list), file=f)
    if Settings.dataset_name == "5k_synthetic_dataset":
        print('oracle test error')
        print(np.average(list_oracle_test_error), np.std(list_oracle_test_error))

        print('n_selected paths, standard error ')
        print(np.average(list_n_selected_paths), np.std(list_n_selected_paths))

    fig_name = data_reader.get_save_location(file_name=fig_name, file_extension=".pdf",
                                             folder_relative_path='results/cross_validation', unique_subfolder=True)
    plot_test_error_vs_iterations(list_of_test_errors, save_fig=save_fig, name_fig=fig_name)


def patience_cross_validation(file_path=None, patience_range=range(5, 100, 5)):
    if file_path is None:
        raise ValueError("file_path cannot be None")
    # list of np array, in which every row of the np array contains the test error for the i-th fold of the cross validation. each element of the list corespond to a different iteration of the cross validation (usually it is just 1 iteration, we use multiple iterations only in the case of synthetic dataset)
    list_test_errors_cross_validation_list: list = data_reader.load_data(directory=file_path)

    list_test_error_sum = [np.sum(list_test_errors_cross_validation_list[i], axis=0) for i in
                           range(len(list_test_errors_cross_validation_list))]

    overfitting_evolution = [[]] * len(list_test_error_sum)
    for i, test_error_sum in enumerate(list_test_error_sum):
        for patience in patience_range:
            overfitting_iteration = early_stopping(test_errors=test_error_sum, patience=int(patience))
            overfitting_evolution[i].append(overfitting_iteration)

    # get average value of each column
    overfitting_evolution = np.mean(np.array(overfitting_evolution), axis=0)
    plot_patience_overfitting_evolution(overfitting_evolution=overfitting_evolution*30, patience_range=patience_range,
                                        saving_location=os.path.dirname(file_path))

    return overfitting_evolution


# uncomment to use the file as a script
if __name__ == '__main__':
    number_of_simulations = 1
    k_folds = 5
    scenario = 3
    patience = 100
    dataset_name = "60k_dataset"
    noise_variance = 0.2
    maximum_number_of_steps = 1800
    save_fig = True
    use_wrapper_boosting = True
    show_settings = True



    cross_validation(number_of_simulations=number_of_simulations, k_folds=k_folds, scenario=scenario, patience=patience,
                     dataset_name=dataset_name, noise_variance=noise_variance,
                     maximum_number_of_steps=maximum_number_of_steps, save_fig=save_fig,
                     use_wrapper_boosting=use_wrapper_boosting, show_settings=show_settings)

    patience_cross_validation(
        file_path="/Users/user/pattern_boosting/results/cross_validation/Xgb_step_1800_max_path_length_6_60k_dataset_gbtree_906/wrapped_boosting/test_errors_cross_validation_list.pkl",
        patience_range=range(5, 300, 5))
