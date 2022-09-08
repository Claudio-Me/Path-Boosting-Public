import numpy as np
from PyAstronomy import pyasl


class BoostingMatrix:
    def __init__(self, matrix: np.ndarray, matrix_header: list):
        assert len(matrix[0]) == len(matrix_header)
        assert isinstance(matrix_header, list)
        assert isinstance(matrix, np.ndarray)

        self.already_selected_columns = set()
        self.matrix = matrix

        # matrix header contains the label_path for each column of the matrix
        self.header = matrix_header

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
        else:
            # very complicate way to add a new column
            self.header.append(header)
            new_matrix = np.zeros((len(self.matrix), len(self.matrix[0])))
            new_matrix[:, -1] = new_column
            new_matrix[:, :-1] = self.matrix
            self.matrix = new_matrix

    def translate_header_to_atom_symbols(self):
        an = pyasl.AtomicNo()
        translated_header = []
        for path in self.header:
            translated_path = []
            for atomic_number in path:
                translated_path.append(an.getElSymbol(atomic_number))
            translated_header.append(translated_path)
        return translated_header
