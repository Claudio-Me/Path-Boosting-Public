from path import Path
from collections import defaultdict


class SelectedPaths:
    def __init__(self):
        self.dictionary = defaultdict(list)

    def is_present(self, path_label):
        return path_label in self.dictionary

    def add_path(self, path_label, path):
        if isinstance(path, Path):
            self.dictionary[path_label].append(path)
        elif hasattr(path, '__iter__'):
            path_objet = Path(path, path[-1])
            self.dictionary[path_label].append(path_objet)



    def get_new_paths_labels_and_add_them_to_the_dictionary(self, path_label, adj_list, node_to_label_dictionary):
        if not self.is_present(path_label):
            return set()
        else:
            paths_to_be_extended = self.dictionary[path_label]
            new_path_labels_new_paths = [old_path.get_next_paths(adj_list, node_to_label_dictionary) for old_path in
                                         paths_to_be_extended]

            new_path_labels_with_no_duplicates = set()
            for label, path in new_path_labels_new_paths:
                self.dictionary[label].append(path)
                new_path_labels_with_no_duplicates.add(label)

            return new_path_labels_with_no_duplicates
