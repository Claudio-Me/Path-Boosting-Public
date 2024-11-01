from collections import defaultdict
from jupiter_notebook_functions import *
from analysis_article.set_default_settings import set_default_settings


# %%


def true_positive_ratio(number_of_simulations=200, synthetic_dataset_scenario=1, noise_variance=0.2,
                        maximum_number_of_steps=None, save_fig=False, show_settings=True):
    settings = set_default_settings()

    settings.noise_variance = noise_variance

    settings.scenario = synthetic_dataset_scenario
    settings.set_scenario(synthetic_dataset_scenario)

    settings.save_analysis = False
    settings.show_analysis = False
    settings.dataset_name = "5k_synthetic_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    settings.generate_new_dataset = True

    if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
        settings.wrapper_boosting = False
    elif synthetic_dataset_scenario == 3:
        settings.wrapper_boosting = True

    if maximum_number_of_steps is None:
        if synthetic_dataset_scenario == 1 or synthetic_dataset_scenario == 2:
            settings.maximum_number_of_steps = 100
        elif synthetic_dataset_scenario == 3:
            settings.maximum_number_of_steps = 350

    final_test_error_vector = []
    final_train_error_vector = []
    missed_paths_counter = []
    n_selected_paths = []
    n_selected_paths_per_iterations = []
    overfitting_iteration = []

    true_positive_ratio_1 = []
    true_positive_ratio_2 = []
    selected_paths_set = set()
    cumulative_paths_importance = defaultdict(float)
    cumulative_times_selected = defaultdict(int)
    counts = defaultdict(int)

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

        for name, times_selected in zip(pattern_boosting.get_boosting_matrix_header(),
                                        pattern_boosting.get_number_of_times_path_has_been_selected()):
            if times_selected > 0:
                cumulative_times_selected[name] += times_selected

        # compute overfitting iteration
        synthetic_dataset = SyntheticDataset(settings=settings)
        overfitting_iteration.append(early_stopping(test_errors=pattern_boosting.test_error, patience=3))
        missed_paths = []
        for target_path in synthetic_dataset.target_paths:
            if target_path not in selected_paths:
                missed_paths.append(target_path)
        missed_paths_counter.append(len(missed_paths))
    averages_importance = {name: cumulative_paths_importance[name] / number_of_simulations for name in
                           cumulative_paths_importance}
    averages_times_selected = {name: cumulative_times_selected[name] / number_of_simulations for name in
                               cumulative_times_selected}

    synthetic_dataset = SyntheticDataset(settings=settings)
    n_target_paths = len(synthetic_dataset.target_paths)

    def plot_tpr_vs_iterations_max_min(true_positive_ratios_per_iteration: list[list[float]], save_fig=True):
        """
        Plots the true positive ratio with standard deviation error bars, max, and min
        against the number of iterations.
        The iterations are inferred based on the length of the true_positive_ratios list.

        :param true_positive_ratios_per_iteration: A list of list containing the TPR values at each iteration for each simulation.
        :type true_positive_ratios_per_iteration: list of list of float
        :returns: None
        """
        # Convert to numpy array for easier manipulation
        np_tpr_per_iteration = np.array(true_positive_ratios_per_iteration)

        # Calculate statistics
        mean_tpr = np.mean(np_tpr_per_iteration, axis=0)
        max_tpr = np.max(np_tpr_per_iteration, axis=0)
        min_tpr = np.min(np_tpr_per_iteration, axis=0)
        std_tpr = np.std(np_tpr_per_iteration, axis=0)  # Calculate the standard deviation

        iterations = list(range(1, len(mean_tpr) + 1))

        plt.figure(figsize=(10, 6))
        # plot standars error:
        # plt.errorbar(iterations, mean_tpr, yerr=std_tpr, fmt='-o', color='b', label='True Positive Ratio (TPR) with Std. Err.', ecolor='red', elinewidth=3, capsize=0)
        plt.errorbar(iterations, mean_tpr, fmt='-o', color='b',
                     label='True Positive Ratio (TPR)', ecolor='red', elinewidth=3, capsize=0)
        plt.fill_between(iterations, min_tpr, max_tpr, color='b', alpha=0.1, label='Min-Max Range')

        plt.xlabel('Iterations')
        plt.ylabel('True Positive Ratio')
        plt.title('True Positive Ratio vs. Iterations with Max, Min')
        plt.legend()
        plt.grid(True)
        if save_fig is True:
            plt.savefig("true_positive_ratio_scenario_" + str(synthetic_dataset_scenario) + ".pdf")
        plt.show()

    if show_settings is True:
        settings.print_principal_values()

    plot_tpr_vs_iterations_max_min(true_positive_ratio_1, save_fig=save_fig)


if __name__ == '__main__':
    number_of_simulations = 200
    noise_variance = 0.2
    maximum_numer_of_steps = None
    save_fig = False
    show_settings = True
    synthetic_dataset_scenario = 3
    true_positive_ratio(number_of_simulations=number_of_simulations,
                        synthetic_dataset_scenario=synthetic_dataset_scenario, noise_variance=noise_variance,
                        maximum_number_of_steps=maximum_numer_of_steps, save_fig=save_fig, show_settings=show_settings)
