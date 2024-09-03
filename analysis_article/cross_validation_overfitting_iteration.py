import sys

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
        fig_name = "test_error_cross_validation_scenario_" + str(Settings.scenario) + ".pdf"
    elif Settings.dataset_name == "5_k_selection_graphs" or Settings.dataset_name == "60k_dataset":
        Settings.generate_new_dataset = False
        fig_name = "test_error_cross_validation.pdf"
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

    # launch cross validation
    for i in range(number_of_simulations):
        print("iteration number ", i)
        print(Settings.maximum_number_of_steps)

        dataset = load_dataset()
        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, Settings.test_size,
                                                                          random_split_seed=Settings.random_split_test_dataset_seed)

        synthetic_dataset = SyntheticDataset()
        oracle_test_error = synthetic_dataset.oracle_model_evaluate(
            graphs_list=test_dataset.get_graphs_list(),
            labels=test_dataset.get_labels())

        list_oracle_test_error.append(oracle_test_error)

        overfitting_iteration, test_error, n_selected_paths = perform_cross_validation(train_dataset, test_dataset,
                                                                                       k=k_folds,
                                                                                       random_seed=Settings.cross_validation_k_fold_seed,
                                                                                       patience=patience)

        list_overfitting_iterations.append(overfitting_iteration)
        list_of_test_errors.append(test_error)
        list_n_selected_paths.append(n_selected_paths)

    if show_settings is True:
        Settings.print_principal_values()

    print("average overfitting iteration:")
    print(np.average(list_overfitting_iterations))

    print("max overfitting iteration")
    print(max(list_overfitting_iterations))

    print("std overfitting iteration:")
    print(np.std(list_overfitting_iterations))

    print("average test error, standard error")
    final_test_error_list = []
    for array_test_error in list_of_test_errors:
        final_test_error_list.append(array_test_error[-1])

    print(np.average(final_test_error_list), np.std(final_test_error_list))
    if Settings.dataset_name == "5k_synthetic_dataset":
        print('oracle test error')
        print(np.average(list_oracle_test_error), np.std(list_oracle_test_error))

        print('n_selected paths, standard error ')
        print(np.average(list_n_selected_paths), np.std(list_n_selected_paths))

    plot_test_error_vs_iterations(list_of_test_errors, save_fig=save_fig, name_fig=fig_name)


# uncomment to use the file as a script
cross_validation(number_of_simulations=1, k_folds=5, scenario=3, patience=100,
                 dataset_name="5_k_selection_graphs", noise_variance=0.2, maximum_number_of_steps=2000, save_fig=True,
                 use_wrapper_boosting=False, show_settings=True)
