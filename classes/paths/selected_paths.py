from classes.paths.path import Path
from collections import defaultdict


class SelectedPaths:
    def __init__(self):
        self.dictionary = defaultdict(list)

    def is_present(self, path_label):
        assert isinstance(path_label, tuple)
        return path_label in self.dictionary

    def add_path(self, path_label, path):
        assert isinstance(path_label, tuple)
        if isinstance(path, Path):
            self.dictionary[path_label].append(path)
        elif hasattr(path, '__iter__'):
            path_objet = Path(path, path[-1])
            self.dictionary[path_label].append(path_objet)

    def get_new_paths_labels_and_add_them_to_the_dictionary(self, path_label, adj_list, node_to_label_dictionary):
        """
        it returns the possible extension of the input path that can be made in the input graph graph
        note: if the input label is not present in the selected paths, an empty set is returned
        note: the choice of having in input adj_list and node_to_label_dictionary instead of the object graph is just to
        avoid self references (this function is usually called inside a graph so it would have to pass self as a
        parameter). To solve this incosistency one could move this function as a method of graph instead of selected_path
        """
        assert isinstance(path_label, tuple)
        if not self.is_present(path_label):
            return set()
        else:
            paths_to_be_extended = self.dictionary[path_label]
            new_path_labels_and_new_paths = [old_path.get_next_paths(adj_list, node_to_label_dictionary, path_label) for
                                             old_path in paths_to_be_extended]
            new_path_labels = []
            new_paths = []
            for item in new_path_labels_and_new_paths:
                assert len(item[0]) == len(item[1])
                new_path_labels.extend(item[0])
                new_paths.extend(item[1])

            new_path_labels_with_no_duplicates = set()
            for i in range(len(new_path_labels)):
                label = new_path_labels[i]
                path = new_paths[i]
                if len(label) > 0:
                    self.add_path(path_label=label, path=path)
                    new_path_labels_with_no_duplicates.add(label)

            return new_path_labels_with_no_duplicates

    def get_number_of_times_path_is_present(self, path_label):
        assert isinstance(path_label, tuple)
        if self.is_present(path_label):
            return len(self.dictionary[path_label])
        else:
            return 0
