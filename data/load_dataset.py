from data.data_reader import load_dataset_from_binary, read_data_from_directory, save_dataset_in_binary_file, save_data
from settings import Settings
from classes.dataset import Dataset
from data.synthetic_dataset import SyntheticDataset
from pathlib import Path


def load_dataset(dataset_name=None):
    if dataset_name is None:
        dataset_name: str = Settings.dataset_name

    # Get the absolute path to the directory where read_data.py is located
    current_dir = Path(__file__).parent.resolve()
    # Construct the absolute path to the dataset file
    dataset_path = current_dir / dataset_name

    if dataset_name == "5_k_selection_graphs":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(directory=current_dir, filename="5_k_selection_graphs")
        else:
            print("Creating 5k dataset")
            if Settings.generate_from_binary_file is False:
                dataset = read_data_from_directory(dataset_path)
            else:
                dataset = load_dataset_from_binary(directory=current_dir, filename="5_k_selection_graphs_original")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset=dataset, directory=current_dir, filename="5_k_selection_graphs")



    elif dataset_name == "60k_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(directory=current_dir, filename="60k_dataset")
        else:
            print("Creating 60k dataset")
            if Settings.generate_from_binary_file is False:
                dataset_path = current_dir / 'dNatQ_graphs'
                dataset = read_data_from_directory(dataset_path)
            else:
                dataset = load_dataset_from_binary(directory=current_dir, filename="60k_dataset_nx_graphs")
            dataset = Dataset(dataset)
            save_dataset_in_binary_file(dataset=dataset, directory=current_dir, filename="60k_dataset")


    elif dataset_name == "5k_synthetic_dataset":
        if Settings.generate_new_dataset is False:
            dataset = load_dataset_from_binary(directory=current_dir, filename="5k_synthetic_dataset")

        else:
            print("Creating a new labels for 5k dataset")
            create_dataset = SyntheticDataset()
            dataset = create_dataset.create_dataset_from_5k_selection_graph(directory=current_dir)
            save_dataset_in_binary_file(dataset=dataset, directory=current_dir, filename="5k_synthetic_dataset")
            writing_directory = current_dir.parent
            writing_directory = writing_directory / "results"
            save_data(create_dataset, filename="synthetic_dataset", directory=writing_directory)
    return dataset


def split_dataset_by_metal_centers(dataset, considered_metal_centers: list = Settings.considered_metal_centers) -> list[
    Dataset]:
    "It returns a list of datasets where dataset in i-th position have all the graphs that have the i-th atom as metal center"
    if not isinstance(dataset, Dataset):
        dataset = Dataset(dataset)
    datasets_list = [[] for i in range(len(considered_metal_centers))]

    # ----------------------------------------------------------------------------------------------------------
    # print("size of dataset: ", asizeof.asizeof(dataset))
    # -----------------------------------------------------------------------------------------------------------

    print("Splitting the dataset")

    for i, graph in enumerate(dataset.get_graphs_list()):
        metal_centers_labels = [graph.node_to_label[metal_center] for metal_center in graph.metal_center]
        for metal_label in metal_centers_labels:
            try:
                index = considered_metal_centers.index(metal_label)
                datasets_list[index].append(graph)
            except:
                print("No metal center found for graph ", i)

    datasets_list = [Dataset(graphs_list) if len(graphs_list) > 0 else None for graphs_list in datasets_list]
    return datasets_list
