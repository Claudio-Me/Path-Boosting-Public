import numpy as np


class BoostingMatrix:
    def __init__(self, matrix, matrix_header):
        assert len(matrix[0]) == len(matrix_header)
        assert isinstance(matrix_header, list)
        assert isinstance(matrix, np.ndarray)

        self.matrix = matrix
        self.header = matrix_header

    def add_column(self, new_column, header):
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
