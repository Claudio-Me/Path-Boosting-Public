import numpy as np
import pandas as pd
from PyAstronomy import pyasl
from typing import List, Tuple
from classes.boosting_matrix import BoostingMatrix
import networkx as nx
from data import data_reader


class ExtendedBoostingMatrix:
    def __int__(self):
        pass
    def extend_boosting_matrix(self, original_boosting_matrix: BoostingMatrix, dataset: list[nx.classes.multigraph.MultiGraph]):
        # we assume the order of observations in boosting matrix is the same as the order in the variable dataset
        data_reader.read_data_from_directory()
        for graph in dataset:

