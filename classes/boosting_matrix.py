import numpy as np
import pandas as pd
from PyAstronomy import pyasl


class BoostingMatrix:
    def __init__(self, matrix: np.ndarray, matrix_header: list, patterns_importance=None):
        assert len(matrix[0]) == len(matrix_header)
        assert isinstance(matrix_header, list)
        assert isinstance(matrix, np.ndarray)

        self.already_selected_columns = set()
        self.matrix = matrix
        self.number_of_times_column_is_selected = np.zeros(len(matrix_header), dtype=int)
        # matrix header contains the label_path for each column of the matrix
        self.header = matrix_header
        if patterns_importance is None:
            self.columns_importance = [0] * len(self.header)
        else:
            self.columns_importance = patterns_importance

    def __get_path_column(self, path: tuple) -> int:
        '''
        It returns none if it is not present
        '''
        header = self.get_header()
        try:
            column = header.index(path)
        except ValueError:
            column = None
        return column


    def get_path_importance(self, path: tuple) -> float:
        column = self.__get_path_column(path)
        if column is None:
            return 0
        else:
            return self.columns_importance[column]

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
            new_cells_number_of_times_column_is_selected = np.zeros(len(header), dtype=int)
        else:
            # very complicate way to add a new column
            self.header.append(header)
            new_matrix = np.zeros((len(self.matrix), len(self.matrix[0])))
            new_matrix[:, -1] = new_column
            new_matrix[:, :-1] = self.matrix
            self.matrix = new_matrix
            new_cells_for_importance_list = [0]
            new_cells_number_of_times_column_is_selected = np.zeros(1, dtype=int)

        self.number_of_times_column_is_selected = np.concatenate([self.number_of_times_column_is_selected,
                                                                  new_cells_number_of_times_column_is_selected])
        self.columns_importance = self.columns_importance + new_cells_for_importance_list

    def update_pattern_importance_of_column(self, column: int, train_error: list, default_value=0):
        '''
        It updates the importance of the column taken in input, and it registers how many times this column has been selected
        that is the number of times column's importance has been update
        :return:
        '''
        self.number_of_times_column_is_selected[column] += 1
        if len(train_error) <= 1:
            self.columns_importance[column] += default_value
        else:
            self.columns_importance[column] += train_error[-2] - train_error[-1]

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

    def get_header(self) -> list:
        return self.header

    def __str__(self):
        string = "Boosting matrix header:\n"
        string = string + str(self.header) + '\n\n'
        string = string + "Selected Paths:\n"
        string = string + str(self.translate_header_to_atom_symbols()) + '\n\n'

        string = string + "number of added paths: " + str(len(self.header)) + '\n'
        string = string + "Number of selected paths " + str(np.count_nonzero(self.columns_importance)) + '\n'

        string = string + "max path length: " + str(self.__max_path_length()) + '\n'
        string = string + "average path length: " + str(self.average_path_length()) + '\n'

        string = string + "Repeated rows in boosting matrix" + str(self.count_repeated_rows()) + "\n"
        string = string + "Different rows in boosting matrix: " + str(self.different_rows()) + "\n\n"

        string = string + "Paths sorted by importance: \n"
        string = string + str(
            sorted(zip(self.columns_importance, self.translate_header_to_atom_symbols()), reverse=True)) + '\n'
        string = string + str(sorted(zip(self.columns_importance, self.get_header()), reverse=True))
        return string

    def count_repeated_rows(self):
        # Create a set to store the rows that have already been seen
        seen_rows = set()

        # Initialize a counter for the number of repeated rows
        repeated_rows = 0

        # Iterate through the rows of the matrix
        for row in self.get_matrix():
            # If the row has already been seen, increment the counter
            if tuple(row) in seen_rows:
                repeated_rows += 1
            # Otherwise, add the row to the set of seen rows
            else:
                seen_rows.add(tuple(row))

        return repeated_rows

    def get_importance_of(self, path):
        if path not in self.header:
            return None
        else:
            index = self.header.index(path)
            return self.columns_importance[index]

    def different_rows(self):
        # Create a set to store the rows that have already been seen
        seen_rows = set()

        # Initialize a counter for the number of repeated rows
        repeated_rows = 0

        # Iterate through the rows of the matrix
        for row in self.get_matrix():
            # If the row has already been seen, increment the counter
            if tuple(row) in seen_rows:
                repeated_rows += 1
            # Otherwise, add the row to the set of seen rows
            else:
                seen_rows.add(tuple(row))

        return len(seen_rows)

    def get_number_of_times_path_has_been_selected(self, path) -> int:
        if isinstance(path, int):
            return self.number_of_times_column_is_selected[path]
        elif isinstance(path, tuple):
            column = self.__get_path_column(path)
            if column is None:
                return 0
            else:
                return self.number_of_times_column_is_selected[column]

    def get_selected_paths(self) -> list:
        header = self.get_header()
        return [header[c] for c in self.already_selected_columns]

    def get_matrix(self):
        return self.matrix

    def get_pandas_matrix(self):
        pd.DataFrame(self.get_matrix(),
                     columns=self.get_header())
