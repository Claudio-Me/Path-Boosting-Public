import xgboost as xgb
from numpy.ma.core import negative
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from classes.dataset import Dataset
from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from classes.boosting_matrix import BoostingMatrix
import networkx as nx
from settings_for_extended_pattern_boosting import SettingsExtendedPatternBoosting
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from classes import analysis
from sklearn.feature_selection import SelectFromModel


class ExtendedPatternBoosting:

    def __init__(self, train_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
                 dict_of_interaction_constraints: dict | None = None,
                 test_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
                 selected_paths: list[tuple[int]] | None = None,
                 settings: SettingsExtendedPatternBoosting | None = None,
                 train_boosting_matrix: np.array = None,
                 test_boosting_matrix: np.array = None):

        # to be able to call ExtendedPatternBoosting methods inside the __init__ method
        super().__init__()

        # they are initialized in the function 'initialize_expanded_boosting_matrix'
        self.dict_of_interaction_constraints: dict | None = None
        self.test_ebm_dataframe: pd.DataFrame | None = None
        self.selected_paths: list[tuple[(int)]] | None = None
        self.train_ebm_dataframe: pd.DataFrame | None = None
        self.train_bm_df: pd.DataFrame | None = None
        self.test_bm_df: pd.DataFrame | None = None
        self.settings: SettingsExtendedPatternBoosting | None = None

        self.initialize_expanded_pattern_boosting(selected_paths=selected_paths,
                                                  train_data=train_data,
                                                  dict_of_interaction_constraints=dict_of_interaction_constraints,
                                                  test_data=test_data,
                                                  settings=settings, train_boosting_matrix=train_boosting_matrix,
                                                  test_boosting_matrix=test_boosting_matrix)

    def train(self, train_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
              dict_of_interaction_constraints: dict | None = None,
              test_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
              selected_paths: list[tuple[int]] | None = None,
              settings: SettingsExtendedPatternBoosting | None = None,
              train_boosting_matrix: np.array = None,
              test_boosting_matrix: np.array = None):

        assert isinstance(train_data, pd.DataFrame) or train_data is None
        assert isinstance(dict_of_interaction_constraints, dict) or dict_of_interaction_constraints is None

        if (train_data is not None) and (dict_of_interaction_constraints is None):
            raise ValueError("dict_of_interaction_constraints should both be given in input, got ",
                             dict_of_interaction_constraints)

        self.initialize_expanded_pattern_boosting(selected_paths=selected_paths,
                                                  train_data=train_data,
                                                  dict_of_interaction_constraints=dict_of_interaction_constraints,
                                                  test_data=test_data,
                                                  settings=settings,
                                                  train_boosting_matrix=train_boosting_matrix,
                                                  test_boosting_matrix=test_boosting_matrix)

        # train, test = train_test_split(self.extended_boosting_matrix_dataframe, test_size=0.2, random_state=0)

        self.settings.main_xgb_parameters["interaction_constraints"] = self.dict_of_interaction_constraints.items
        self.settings.main_xgb_parameters['n_estimators'] = 1

        x_df_train, y_train = self.split_target_from_dataframe(self.train_ebm_dataframe)
        x_df_test, y_test = self.split_target_from_dataframe(self.test_ebm_dataframe)

        xgb_model = xgb.XGBRegressor(**self.settings.main_xgb_parameters)
        evallist = [(x_df_train, y_train), (x_df_test, y_test)]

        for n_iteration in range(self.settings.n_estimators):

            if n_iteration == 0:
                best_path = self.__find_best_path(self.train_bm_df, y_train)
                zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
                    x_df=x_df_train, y=y_train, path=best_path,
                    dict_of_interaction_constraints=self.dict_of_interaction_constraints)
                xgb_model = xgb_model.fit(zeroed_x_df, zeroed_y, eval_set=evallist)

            else:
                # evals_results=xgb_model.eval_result()
                # last_training_error=evals_results['validation_0'][self.settings.main_xgb_parameters['eval_metric']][-1]

                self.settings.main_xgb_parameters['n_estimators'] = n_iteration + 1
                new_xgb_model = xgb.XGBRegressor(**self.settings.main_xgb_parameters)

                predictions = xgb_model.predict(x_df_train)
                negative_gradient = self.neg_gradient(y=y_train.to_numpy(), y_hat=np.array(predictions))

                best_path = self.__find_best_path(self.train_bm_df, pd.Series(negative_gradient))

                #--------------------------------------------------------
                print("best path")
                print(best_path)
                #------------------------------------------------------

                zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
                    x_df=x_df_train, y=y_train, path=best_path,
                    dict_of_interaction_constraints=self.dict_of_interaction_constraints)
                xgb_model = new_xgb_model.fit(zeroed_x_df, zeroed_y, eval_set=evallist, xgb_model=xgb_model)

        if self.settings.plot_analysis is True:
            ExtendedPatternBoosting.training_results(bst=xgb_model, X_test=x_df_test, y_test=y_test)

    @staticmethod
    def split_target_from_dataframe(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        # it splits the dataframe extracting the target column
        y_target = df['target']
        x_df = df.drop(['target'], axis=1, inplace=False)
        return x_df, y_target

    @staticmethod
    def __find_best_path(x_df: pd.DataFrame, y_target: pd.Series) -> tuple:
        # it returns the best column to be selected, chosen by running xgb on boosting matrix with zeroes and ones

        print(f"{y_target.head()=}")

        choose_column_xgb_parameters = SettingsExtendedPatternBoosting().choose_column_xgb_parameters
        xgb_local_model = xgb.XGBRegressor(**choose_column_xgb_parameters)

        # alternative way to select best column using sklearn
        #selector=SelectFromModel(xgb_local_model,threshold=-np.inf, max_features=1, prefit=False).fit(x_df, y_target)
        #best_path=selector.get_feature_names_out(x_df.columns)[0]

        xgb_local_model = xgb_local_model.fit(x_df, y_target)
        selected_column = np.argsort(xgb_local_model.feature_importances_)
        print(f"{xgb_local_model.feature_importances_=}")
        selected_column = selected_column[-1]

        best_path = x_df.columns[selected_column]

        return best_path

    def initialize_expanded_pattern_boosting(self, selected_paths: list[tuple[int]] | None = None,
                                             train_data: pd.DataFrame | list[
                                                 nx.classes.multigraph.MultiGraph] | None = None,
                                             dict_of_interaction_constraints: dict | None = None,
                                             test_data: pd.DataFrame | list[
                                                 nx.classes.multigraph.MultiGraph] | None = None,
                                             settings: SettingsExtendedPatternBoosting | None = None,
                                             train_boosting_matrix: np.array = None,
                                             test_boosting_matrix: np.array = None
                                             ):

        if train_data is None:
            return
        if settings is not None:
            self.settings = settings

        if hasattr(selected_paths, '__iter__'):
            self.selected_paths = selected_paths

        if isinstance(train_data, pd.DataFrame):
            assert "target" in train_data.columns
            self.train_ebm_dataframe = train_data
        elif isinstance(train_data, list):
            self.train_ebm_dataframe = ExtendedBoostingMatrix.create_extend_boosting_matrix_for(
                selected_paths=selected_paths, list_graphs_nx=train_data)

        if dict_of_interaction_constraints is None and self.dict_of_interaction_constraints is None:
            assert isinstance(train_data, list)
            self.dict_of_interaction_constraints = ExtendedBoostingMatrix.get_features_interaction_constraints(
                list_graphs_nx=train_data, selected_paths=self.selected_paths)
        elif dict_of_interaction_constraints is not None:
            self.dict_of_interaction_constraints = dict_of_interaction_constraints

        if train_boosting_matrix is not None:
            self.train_bm_df = pd.DataFrame(train_boosting_matrix)
            self.train_bm_df.columns = self.selected_paths
        elif isinstance(train_data, list):
            self.train_bm_df = ExtendedBoostingMatrix.create_boosting_matrix(selected_paths=self.selected_paths,
                                                                             list_graphs_nx=train_data)
        else:
            raise Exception(
                "impossible to create boosting matrix for train data, provide boosting matrix or list of graph")

        self.set_up_test_data(test_data=test_data, test_boosting_matrix=test_boosting_matrix)

    def set_up_test_data(self, test_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None,
                         test_boosting_matrix: np.array = None):
        if test_data is None:
            return
        if isinstance(test_data, pd.DataFrame):
            self.test_ebm_dataframe = test_data
        elif hasattr(test_data, '__iter__'):
            if isinstance(test_data[0], nx.classes.multigraph.MultiGraph):
                self.test_ebm_dataframe = ExtendedBoostingMatrix.create_extend_boosting_matrix_for(
                    selected_paths=self.selected_paths,
                    list_graphs_nx=test_data)
            else:
                raise TypeError("test_data type unrecognized", test_data)
        else:
            raise TypeError("test_data type unrecognized", test_data)

        if test_boosting_matrix is not None:
            self.train_bm_df = pd.DataFrame(test_boosting_matrix)
            self.train_bm_df.columns = self.selected_paths
        elif isinstance(test_data, list):
            self.train_bm_df = ExtendedBoostingMatrix.create_boosting_matrix(selected_paths=self.selected_paths,
                                                                             list_graphs_nx=test_data)
        else:
            raise Exception(
                "impossible to create boosting matrix for test data, provide boosting matrix or list of graph")

    @staticmethod
    def training_results(bst, X_test, y_test):

        predictions = bst.predict(X_test)

        def calculate_r2_score(y_true, y_pred):
            # Calculate the sum of squares of residuals
            ss_res = np.sum((y_true - y_pred) ** 2)

            # Calculate the total sum of squares
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

            # Calculate the R^2 score
            r2_score = 1 - (ss_res / ss_tot)
            return r2_score

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_true=y_test, y_pred=predictions)
        my_r2 = calculate_r2_score(y_true=y_test, y_pred=predictions)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-Squared:", r2)
        print("My R-Squared:", my_r2)

        # Plotting the feature importance
        # xgb.plot_importance(bst)
        # plt.show()

        analysis.density_scatterplot(y_test, predictions, show_fig=True, save_fig=True)

        # Optional: Return the metrics if you need them for further analysis
        return {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }

    @staticmethod
    def neg_gradient(y, y_hat):
        return (y - y_hat)
