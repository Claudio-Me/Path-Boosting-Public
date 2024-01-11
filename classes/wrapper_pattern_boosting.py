# Takes as input a lyst of parrent boosting models and "merge" them into one
from settings import Settings
from classes.dataset import Dataset
from classes.graph import GraphPB
from data.load_dataset import split_dataset_by_metal_centers
from classes.pattern_boosting import PatternBoosting
from multiprocessing.dummy import Pool as ThreadPool
import functools


class WrapperPatternBoosting:
    def __init__(self, pattern_boosting_list: list = None, metal_center_list: list = Settings.considered_metal_centers):
        if pattern_boosting_list is None:
            pattern_boosting_list = [PatternBoosting() for _ in range(len(metal_center_list))]

        if len(pattern_boosting_list) != len(metal_center_list):
            raise ValueError("not enough models for each metal center")
        self.pattern_boosting_models_list = pattern_boosting_list
        self.metal_center_list = metal_center_list

    def predict(self, graphs_list, boosting_matrix_matrix=None):
        # If a graph has more metal centers ,the final prediction will be just the average between the different models
        if isinstance(graphs_list, GraphPB):
            graphs_list = [graphs_list]
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

    @staticmethod
    def __train_pattern_boosting(pattern_boosting_model: PatternBoosting, train_dataset, test_dataset):
        pattern_boosting_model.training(train_dataset, test_dataset)
        return pattern_boosting_model

    def train(self, train_datasets_list, test_datasets_list=None):

        if test_datasets_list is None:
            test_datasets_list = [None for _ in range(len(train_datasets_list))]

        # Paralelization
        # ------------------------------------------------------------------------------------------------------------


        pool = ThreadPool(min(10, len(Settings.considered_metal_centers)))
        array_of_outputs = pool.map(
            functools.partial(self.__train_patern_boosting), self.pattern_boosting_models_list, train_datasets_list,
            test_datasets_list)
        # -------------------------------------------------------------------------------------------------------------
        return array_of_outputs
