import pandas as pd
from classes.extended_boosting_matrix import ExtendedBoostingMatrix

# Sample DataFrame and selected_paths (replace these with your actual DataFrame and list)
df_example = pd.DataFrame({
    "(1,2,)_some_name": [1, 2, 3],
    "(1,)_another_name": [4, 5, 6],
    "(2,3,)_different_name": [7, 8, 9],
    "(1,2,3,4)_name": [10, 11, 12]
})
selected_paths_example = [(1, 2, 4, 5), (1, 1, 2, 4, 5)]

extended_boosting_matrix = ExtendedBoostingMatrix()

extended_boosting_matrix.df=df_example


# Using the function to get the dictionary
tuple_to_column_indices_dict = extended_boosting_matrix.associate_paths_to_columns(selected_paths_example)

# Print true if the result is correct to verify the result
expected_result={(1, 2, 4, 5): [0, 1], (1, 1, 2, 4, 5): [1]}

res = all((tuple_to_column_indices_dict.get(k) == v for k, v in expected_result.items()))
print(res)
