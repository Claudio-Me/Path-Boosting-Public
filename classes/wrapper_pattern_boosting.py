# Takes as input a lyst of parrent boosting models and "merge" them into one
from settings import Settings
from classes.dataset import Dataset
from classes.graph import GraphPB
from data.load_dataset import split_dataset_by_metal_centers


class WrapperPatternBoosting:
    def __init__(self, pattern_boosting_list: list, metal_center_list: list = Settings.considered_metal_centers):
        if len(pattern_boosting_list) != len(metal_center_list):
            raise ValueError("not enough models for each metal center")
        self.pattern_boosting_models_list = pattern_boosting_list
        self.metal_center_list = metal_center_list

    def predict(self, graphs_list, boosting_matrix_matrix=None):
        # If a graph has more metal centers ,the final prediction will be just the average between the different models
        if isinstance(graphs_list, GraphPB):
            graphs_list=[graphs_list]
        if isinstance(graphs_list, Dataset):
            graphs_list = graphs_list.get_graphs_list()
        prediction = [] * len(graphs_list)
        for i, graph in enumerate(graphs_list):
            metal_centers_labels = [graph.node_to_label[metal_center] for metal_center in graph.metal_center]
            graph_prediction = 0
            counter = 0
            for metal_label in metal_centers_labels:
                try:
                    index = self.metal_center_list.index(metal_label)
                    graph_prediction = graph_prediction + self.pattern_boosting_models_list[index].predict(graph)
                    counter += 1
                except:
                    pass

            graph_prediction = graph_prediction / counter
            prediction[i] = graph_prediction
        return prediction