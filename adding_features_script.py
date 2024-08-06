from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from data import data_reader
from classes.dataset import Dataset
import matplotlib as plt
import pandas as pd
import numpy as np

# read the original dataset:
# dataset_original_graphs = data_reader.read_data_from_directory("data/5k-selection-graphs")

# data_reader.save_dataset_in_binary_file(dataset_original_graphs, filename="5_k_selection_graphs_original")

dataset_original_graphs = data_reader.load_dataset_from_binary(filename="5_k_selection_graphs_original")

# load the trained model that contains the boosting matrix


directory = "/Users/popcorn/PycharmProjects/pattern_boosting/results/Xgb_step_300_max_path_length_101_5_k_selection_graphs/wrapped_boosting"

wrapper_pattern_boosting = data_reader.load_data(directory=directory, filename="wrapper_pattern_boosting")

selected_paths = wrapper_pattern_boosting.get_selected_paths()

extended_boosting_matrix = ExtendedBoostingMatrix()
extended_boosting_matrix.create_extend_boosting_matrix(selected_paths=selected_paths,
                                                       dataset=dataset_original_graphs)
extended_boosting_matrix.df.loc[:, : 10000] = np.nan
print(extended_boosting_matrix.df.dtypes)
new_df = extended_boosting_matrix.df.astype(pd.SparseDtype(float, fill_value=np.nan))
print("sparse type")
print(new_df.dtypes)

from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

# First, convert the sparse DataFrame to a scipy sparse matrix
sparse_matrix = csc_matrix(extended_boosting_matrix.df)

plt.figure(figsize=(10, 10))
plt.spy(sparse_matrix, markersize=1)
plt.show()





# Get the sparsity pattern for each SparseDtype column
locations = []
for column in new_df.columns:
    sparse_series = new_df[column]
    if pd.api.types.is_sparse(sparse_series):
        sparse_array = sparse_series.array
        # Get the indices of the non-zero entries
        non_zero_indices = sparse_array.sp_index.to_int_index().indices
        col_index = new_df.columns.get_loc(column)
        locations.extend(zip(non_zero_indices, [col_index] * len(non_zero_indices)))

# Unpack the locations to separate lists of rows and columns
rows, cols = zip(*locations)

plt.figure(figsize=(10, 6))
plt.scatter(cols, rows, alpha=0.5)
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.gca().invert_yaxis()  # Invert the y-axis to match the matrix representation
plt.show()






non_missing_mask = extended_boosting_matrix.df.notna()

# Stack to get indices of non-missing values
non_missing_indices = non_missing_mask.stack()

# The indices will be a MultiIndex where the first level is the row index and second level is the column index
rows, cols = zip(*non_missing_indices.index)

plt.figure(figsize=(10, 6))
plt.scatter(cols, rows, alpha=0.5)
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.gca().invert_yaxis()  # Invert the y-axis to match the matrix representation
plt.show()
