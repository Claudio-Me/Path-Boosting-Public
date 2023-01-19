import numpy as np
from PyAstronomy import pyasl


class BoostingMatrix:
    def __init__(self, matrix: np.ndarray, matrix_header: list, patterns_importance=None):
        assert len(matrix[0]) == len(matrix_header)
        assert isinstance(matrix_header, list)
        assert isinstance(matrix, np.ndarray)

        self.already_selected_columns = set()
        self.matrix = matrix

        # matrix header contains the label_path for each column of the matrix
        self.header = matrix_header
        if patterns_importance is None:
            self.patterns_importance = [0] * len(self.header)
        else:
            self.patterns_importance = patterns_importance

    def add_column(self, new_column, header):
        """"
        given a list of columns and their header adds them to the main matrix
        """
        assert len(new_column) == len(self.matrix)
        new_column = np.array(new_column)

        if isinstance(new_column[0], np.ndarray):
            # we have multiple columns to add
            assert len(new_column[0]) == len(header)
            self.header = self.header + list(header)

            self.matrix = np.concatenate((self.matrix, new_column), axis=1)

            new_cells_for_importance_list = [0] * len(header)
        else:
            # very complicate way to add a new column
            self.header.append(header)
            new_matrix = np.zeros((len(self.matrix), len(self.matrix[0])))
            new_matrix[:, -1] = new_column
            new_matrix[:, :-1] = self.matrix
            self.matrix = new_matrix
            new_cells_for_importance_list = [0]

        self.patterns_importance = self.patterns_importance + new_cells_for_importance_list

    def update_pattern_importance_of_column(self, column: int, train_error: list, default_value=0):
        if len(train_error) <= 1:
            self.patterns_importance[column] += default_value
        else:
            self.patterns_importance[column] += train_error[-2] - train_error[-1]

    def translate_header_to_atom_symbols(self):
        an = pyasl.AtomicNo()
        translated_header = []
        for path in self.header:
            translated_path = []
            for atomic_number in path:
                translated_path.append(an.getElSymbol(atomic_number))
            translated_header.append(translated_path)
        return translated_header

    def average_path_length(self):
        average_path_length = 0

        for path in self.header:
            average_path_length = average_path_length + len(path)
        average_path_length = average_path_length / len(self.header)
        return average_path_length

    def __max_path_length(self):
        max_length = 0

        for path in self.header:
            if len(path) > max_length:
                max_length = len(path)
        return max_length

    def get_header(self):
        return self.header

    def __str__(self):
        string = "Boosting matrix header:\n"
        string = string + str(self.header) + '\n\n'
        string = string + "Selected Paths:\n"
        string = string + str(self.translate_header_to_atom_symbols()) + '\n\n'

        string = string + "number of added paths: " + str(len(self.header)) + '\n'
        string = string + "Number of selected paths " + str(np.count_nonzero(self.patterns_importance)) + '\n'

        string = string + "max path length: " + str(self.__max_path_length()) + '\n'
        string = string + "average path length: " + str(self.average_path_length()) + '\n\n'

        string = string + "Paths sorted by importance: \n"
        string = string + str(
            sorted(zip(self.patterns_importance, self.translate_header_to_atom_symbols()), reverse=True)) + '\n'
        string = string + str(sorted(zip(self.patterns_importance, self.get_header()), reverse=True))
        return string
