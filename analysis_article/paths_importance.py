from collections import defaultdict
import sys

from scipy.ndimage import maximum

sys.path.insert(0, "../")
from jupiter_notebook_functions import *
from analysis_article.set_default_settings import set_default_settings


def paths_importance_analysis(dataset_name, number_of_simulations=200, synthetic_dataset_scenario=1, noise_variance=0.2,
                              maximum_number_of_steps=None, update_features_importance_by_comparison=True,
                              show_settings=True):
    settings = set_default_settings()

    settings.noise_variance = noise_variance

    settings.scenario = synthetic_dataset_scenario
    settings.set_scenario(synthetic_dataset_scenario)

    settings.update_features_importance_by_comparison = update_features_importance_by_comparison

    settings.save_analysis = False
    settings.show_analysis = False
    settings.dataset_name = dataset_name  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    settings.generate_new_dataset = True

    if dataset_name == '5k_synthetic_dataset':
        if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
            settings.wrapper_boosting = False
        elif synthetic_dataset_scenario == 3:
            settings.wrapper_boosting = True

        if maximum_number_of_steps is None:
            if synthetic_dataset_scenario == 1:
                settings.maximum_number_of_steps = 28
            if synthetic_dataset_scenario == 2:
                settings.maximum_number_of_steps = 83
            elif synthetic_dataset_scenario == 3:
                settings.maximum_number_of_steps = 300


    else:
        settings.wrapper_boosting = True

    final_test_error_vector = []
    final_train_error_vector = []
    missed_paths_counter = []
    n_selected_paths = []
    n_selected_paths_per_iterations = []
    overfitting_iteration = []
    true_positive_ratio_1 = []
    selected_paths_set = set()
    cumulative_paths_importance = defaultdict(float)
    cumulative_times_selected = defaultdict(int)
    counts = defaultdict(int)
    dictionary_paths_importance_stored_in_lists = defaultdict(list)
    dictionary_n_times_selected_stored_in_lists = defaultdict(list)

    for i in range(number_of_simulations):
        print("i")
        print(i)
        dataset = load_dataset(settings= settings)

        train_dataset, test_dataset = data_reader.split_training_and_test(dataset, settings.test_size,
                                                                          random_split_seed=settings.random_split_test_dataset_seed)

        # pattern boosting
        pattern_boosting = PatternBoosting(settings=settings)
        pattern_boosting.training(train_dataset, test_dataset)
        final_test_error = pattern_boosting.test_error[-1]
        final_train_error = pattern_boosting.train_error[-1]
        final_test_error_vector.append(final_test_error)
        final_train_error_vector.append(final_train_error)
        n_selected_paths_per_iterations.append(pattern_boosting.n_selected_paths)
        true_positive_ratio_1.append(pattern_boosting.true_positive_ratio_1)
        selected_paths = pattern_boosting.get_selected_paths_in_boosting_matrix()
        n_selected_paths.append(len(selected_paths))
        # compute number of times a path is selected and average importance
        patterns_importance = pattern_boosting.get_boosting_matrix_normalized_columns_importance_values()

        for name, value in zip(pattern_boosting.get_boosting_matrix_header(), patterns_importance):
            if value > 0.0:
                cumulative_paths_importance[name] += value
                counts[name] += 1
                dictionary_paths_importance_stored_in_lists[name].append(value)

        for name, times_selected in zip(pattern_boosting.get_boosting_matrix_header(),
                                        pattern_boosting.get_number_of_times_path_has_been_selected()):
            if times_selected > 0:
                cumulative_times_selected[name] += times_selected
                dictionary_n_times_selected_stored_in_lists[name].append(times_selected)

        # compute overfitting iteration
        synthetic_dataset = SyntheticDataset(settings=settings)
        overfitting_iteration.append(early_stopping(test_errors=pattern_boosting.test_error, patience=3))
        missed_paths = []
        for target_path in synthetic_dataset.target_paths:
            if target_path not in selected_paths:
                missed_paths.append(target_path)
        missed_paths_counter.append(len(missed_paths))

    # add zeroes to every path list such that the length of the list for each path is equal to number of simulations
    for name in dictionary_paths_importance_stored_in_lists:
        if len(dictionary_paths_importance_stored_in_lists[name]) < number_of_simulations:
            zeroes = [0] * (number_of_simulations - len(dictionary_paths_importance_stored_in_lists[name]))
            dictionary_paths_importance_stored_in_lists[name] = dictionary_paths_importance_stored_in_lists[
                                                                    name] + zeroes

    for name in dictionary_n_times_selected_stored_in_lists:
        if len(dictionary_n_times_selected_stored_in_lists[name]) < number_of_simulations:
            zeroes = [0] * (number_of_simulations - len(dictionary_n_times_selected_stored_in_lists[name]))
            dictionary_n_times_selected_stored_in_lists[name] = dictionary_n_times_selected_stored_in_lists[
                                                                    name] + zeroes

    averages_importance = {name: (cumulative_paths_importance[name] / number_of_simulations,
                                  np.std(dictionary_paths_importance_stored_in_lists[name])) for name in
                           cumulative_paths_importance}
    averages_times_selected = {name: (
        cumulative_times_selected[name] / number_of_simulations,
        np.std(dictionary_n_times_selected_stored_in_lists[name]))
        for name in
        cumulative_times_selected}

    if show_settings is True:
        settings.print_principal_values()

    # Print averages values of results over synthetic dataset
    print("Averages importances")
    print_dict_sorted_by_values(averages_importance)

    print("average number of times selected")
    print_dict_sorted_by_values(averages_times_selected)

    print("final_test_error_vector")
    print(np.average(final_test_error_vector), np.std(final_test_error_vector))
    print("final_train_error_vector")
    print(np.average(final_train_error_vector), np.std(final_train_error_vector))
    print("n_selected_paths")
    print(np.average(n_selected_paths), np.std(n_selected_paths))

    # print("n_selected_paths_vector")
    # print(np.average(n_selected_paths_per_iterations, axis=0), np.std(n_selected_paths_per_iterations))

    print("overfitting_iteration")
    print(np.average(overfitting_iteration), np.std(overfitting_iteration))

    print("coefficients for the synthetic dataset")
    synthetic_dataset = SyntheticDataset(settings=settings)
    print(synthetic_dataset.target_paths)
    print(synthetic_dataset.coefficients)

    avg_selected_paths_per_iterations = np.average(n_selected_paths_per_iterations, axis=0)
    synthetic_dataset = SyntheticDataset(settings=settings)
    n_target_paths = len(synthetic_dataset.target_paths)


if __name__ == '__main__':
    dataset_name = "5k_synthetic_dataset"  # "5k_synthetic_dataset"  "5_k_selection_graphs"  "60k_dataset"
    synthetic_dataset_scenario = 3  # used only in the case dataset_name is "5k_synthetic_dataset"
    number_of_simulations = 200
    noise_variance = 0.2
    maximum_number_of_steps = None
    update_features_importance_by_comparison = True
    show_settings = True
    paths_importance_analysis(dataset_name=dataset_name, number_of_simulations=number_of_simulations,
                              synthetic_dataset_scenario=synthetic_dataset_scenario,
                              noise_variance=noise_variance, maximum_number_of_steps=maximum_number_of_steps,
                              update_features_importance_by_comparison=update_features_importance_by_comparison,
                              show_settings=show_settings)
