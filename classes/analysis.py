import matplotlib.pyplot as plt
import seaborn as sns


def compare_performances_on_synthetic_dataset(test_model_preds, oracle_model_preds, true_values, dataset):
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
    :return: None
    """
    #plt.style.use('classic')
    plt.figure(figsize=(12, 6))

    sns.kdeplot(test_model_preds, color='blue', label='Test Model')
    sns.kdeplot(oracle_model_preds, color='red', label='Oracle Model')
    sns.kdeplot(true_values, color='green', label='True Values')

    plt.legend()
    plt.xlabel('Predicted/True Value')
    plt.title(f'Model vs Oracle Model vs True Values on {dataset}')  # Added dataset to the title
    plt.grid(True)



    plt.show()



import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def plot_error_evolution(error_list: list, dataset: str):
    """
    This function takes a list of error values and a dataset name as input.
    It creates a line plot of the error evolution over time using the matplotlib library.

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
    ax.plot(iterations, error_list, marker='o')

    # Inverse log scale for the y-axis
    ax.set_yscale("log")

    # Add title and labels
    ax.set_title(dataset + ' Error')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')

    # Show the plot
    plt.show()
