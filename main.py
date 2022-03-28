from testing.testing import Testing

from pattern_boosting import PatternBoosting

import data.data_reader as dt
import networkx as nx
import numpy as np

if __name__ == '__main__':
    Testing()
    LALMER_graph = dt.read_data("LALMER.gml")
    OREDIA_graph = dt.read_data("OREDIA.gml")
    dataset = [LALMER_graph, OREDIA_graph]
    PatternBoosting(dataset)
