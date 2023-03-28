from data.data_reader import load_dataset_from_binary, read_data_from_directory,save_dataset_in_binary_file, save_data
from settings import Settings
from classes.dataset import Dataset
from data.synthetic_dataset import SyntheticDataset

def load_dataset():
    if Settings.dataset_name == "5_k_selection_graphs":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="5_k_selection_graphs")
        else:
            print("Creating 5k dataset")
            dataset = read_data_from_directory("data/5k-selection-graphs")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset, filename="5_k_selection_graphs")



    elif Settings.dataset_name == "60k_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="60k_dataset")
        else:
            print("Creating 60k dataset")
            dataset = read_data_from_directory("data/dNatQ_graphs")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset, filename="60k_dataset")


    elif Settings.dataset_name == "5k_synthetic_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(filename="5k_synthetic_dataset")

        else:
            print("Creating a new labels for 5k dataset")
            create_dataset = SyntheticDataset()
            dataset = create_dataset.create_dataset_from_5k_selection_graph()
            save_dataset_in_binary_file(dataset, filename="5k_synthetic_dataset")

            save_data(create_dataset, filename="synthetic_dataset", directory="results")
    return dataset