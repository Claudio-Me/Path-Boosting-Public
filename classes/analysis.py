import matplotlib.pyplot as plt
import seaborn as sns
from data import data_reader


def compare_performances_on_synthetic_dataset(test_model_preds, oracle_model_preds, true_values, dataset: str,
                                              save: bool = False, show: bool = False):
    """
    Function to compare and plot predictions of two models and true values on a synthetic dataset.

    :param test_model_preds: list
        Test model predictions, list of numeric values.
    :param oracle_model_preds: list
        Oracle model predictions, list of numeric values.
    :param true_values: list
        True values, list of numeric values.
    :param dataset: string
        The name of the dataset.
    :param show: True to show the plot
    :param save: True to save the plot
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.kdeplot(test_model_preds, color='blue', label='Test Model', ax=ax)
    sns.kdeplot(oracle_model_preds, color='red', label='Oracle Model', ax=ax)
    sns.kdeplot(true_values, color='green', label='True Values', ax=ax)

    ax.legend()
    ax.set_xlabel('Predicted/True Value')
    ax.set_title(f'Model vs Oracle Model vs True Values on {dataset}')  # Added dataset to the title
    ax.grid(True)

    # Show the plot
    if show is True:
        plt.show()

    if save is True:
        saving_location = data_reader.get_save_location(file_name=dataset + '_Model_vs_Oracle',
                                                        file_extension=".png",
                                                        folder_relative_path='results', unique_subfolder=True)


        fig.savefig(saving_location)


def plot_error_evolution(error_list: list, dataset: str, save: bool = False, show: bool = False):
    """
    This function takes a list of error values and a dataset name as input.
    It creates a line plot of the error evolution over time using the matplotlib library.

    :param show: True to show the plot
    :param save: True to save the plot
    :param error_list: A list of error values representing the evolution
                       of the error over various algorithm iterations.
    :type error_list: list
    :param dataset: The name of the dataset used in the iteration.
    :type dataset: str

    :return: None
    """
    # Create a list with the iteration numbers
    iterations = list(range(1, len(error_list) + 1))
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, error_list)

    # Inverse log scale for the y-axis
    ax.set_yscale("log")

    # Add title and labels
    ax.set_title(dataset + ' Error')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')

    # Show the plot
    if show is True:
        fig.show()

    if save is True:
        saving_location = data_reader.get_save_location(file_name=dataset + "_error_evolution", file_extension=".png",
                                                        folder_relative_path='results', unique_subfolder=True)

        fig.savefig(saving_location)
