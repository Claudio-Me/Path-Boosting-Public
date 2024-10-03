import xgboost as xgb
import numpy as np
import pandas as pd
from classes.extended_boosting_matrix import ExtendedBoostingMatrix
import networkx as nx

from classes.extended_path_boosting.models_for_extendded_path_boosting.xgb_model_for_extended_path_boosting import \
    XgbModelForExtendedPathBoosting

from classes.extended_path_boosting.models_for_extendded_path_boosting.addititive_xgb_for_externded_path_bosting import \
    AdditiveXgbForExtendedPathBosting
from settings_for_extended_pattern_boosting import SettingsExtendedPatternBoosting
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from classes import analysis
import ast


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
        self.paths_selected_by_epb = []
        self.model: XgbModelForExtendedPathBoosting | AdditiveXgbForExtendedPathBosting | None = None
        self.train_error = []
        self.test_error = []

        self.initialize_expanded_pattern_boosting(
            selected_paths=selected_paths,
            train_data=train_data,
            dict_of_interaction_constraints=dict_of_interaction_constraints,
            test_data=test_data,
            settings=settings,train_boosting_matrix=train_boosting_matrix,
            test_boosting_matrix=test_boosting_matrix
        )

    def train(self, train_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
              dict_of_interaction_constraints: dict | None = None,
              test_data: pd.DataFrame | list[nx.classes.multigraph.MultiGraph] | None = None,
              selected_paths: list[tuple[int]] | None = None,
              settings: SettingsExtendedPatternBoosting | None = None,
              train_boosting_matrix: np.array = None,
              test_boosting_matrix: np.array = None
              ):

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
                                                  test_boosting_matrix=test_boosting_matrix
                                                  )

        # train, test = train_test_split(self.extended_boosting_matrix_dataframe, test_size=0.2, random_state=0)

        self.settings.main_xgb_parameters["interaction_constraints"] = self.dict_of_interaction_constraints.items
        self.settings.main_xgb_parameters['n_estimators'] = 1

        x_df_train, y_train = self.split_target_from_dataframe(self.train_ebm_dataframe)
        if self.test_ebm_dataframe is not None:
            x_df_test, y_test = self.split_target_from_dataframe(self.test_ebm_dataframe)
            evallist = [(x_df_train, y_train), (x_df_test, y_test)]
        else:
            evallist = [(x_df_train, y_train)]

        for n_iteration in range(self.settings.n_estimators):
            print("Iteration ", n_iteration + 1)

            if n_iteration == 0:
                best_path = self.__find_best_path(self.train_bm_df, y_train)
                zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
                    x_df=x_df_train, y=y_train, path=best_path,
                    dict_of_interaction_constraints=self.dict_of_interaction_constraints)
                self.model = self.model.fit(zeroed_x_df, zeroed_y, eval_set=evallist)
                # -------------------------------------------------------------------
                # TODO delete this
                # delete me
                # we compute the error of the predictor
                predictions = self.model.predict(zeroed_x_df)
                mse = mean_squared_error(y_true=zeroed_y, y_pred=predictions)
                print(f"predictor {mse=}")
                # -------------------------------------------------------------------

                # self.xgb_model.get_booster().get_score(importance_type='weight')
                # self.xgb_model.get_booster().get_dump()

            else:
                # evals_results=self.xgb_model.eval_result()
                # last_training_error=evals_results['validation_0'][self.settings.main_xgb_parameters['eval_metric']][-1]

                predictions = self.model.predict(x_df_train)
                negative_gradient = self.neg_gradient(y=y_train.to_numpy(), y_hat=np.array(predictions))

                best_path = self.__find_best_path(self.train_bm_df, pd.Series(negative_gradient))

                zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
                    x_df=x_df_train, y=y_train, path=best_path,
                    dict_of_interaction_constraints=self.dict_of_interaction_constraints)

                # self.settings.main_xgb_parameters['n_estimators'] = n_iteration + 1
                y_zero_hat = self.model.predict(zeroed_x_df)
                negative_gradient_for_zeroed_matrix = pd.Series(negative_gradient).loc[zeroed_y.index]
                new_target = y_zero_hat + negative_gradient_for_zeroed_matrix
                self.model.fit(X=zeroed_x_df, y=new_target, eval_set=evallist)
                # self.model.fit(zeroed_x_df, zeroed_y, eval_set=evallist)
                # save train and test error to a list to have its evolution

                # -------------------------------------------------------------------
                # TODO delete this
                # delete me
                # we compute the error of the predictor
                predictions = self.model.predict(x_df_train)
                mse = mean_squared_error(y_true=y_train, y_pred=predictions)
                print(f"predictor {mse=}")
                # -------------------------------------------------------------------

            print("best path")
            print(best_path)
            # self.xgb_model.get_booster().get_score(importance_type='weight')
            # self.xgb_model.get_booster().get_dump()
            if self.settings.show_tree is True:
                self.model.plot_tree()

            self.paths_selected_by_epb.append(best_path)
        if self.settings.plot_analysis is True and self.test_ebm_dataframe is not None:
            self.__training_results(X_test=x_df_test, y_test=y_test)

    @staticmethod
    def split_target_from_dataframe(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        # it splits the dataframe extracting the target column
        y_target = df['target']
        x_df = df.drop(['target'], axis=1, inplace=False)
        return x_df, y_target

    @staticmethod
    def __find_best_path(x_df: pd.DataFrame, y_target: pd.Series) -> tuple:
        # it returns the best column to be selected, chosen by running xgb on boosting matrix with zeroes and ones

        # x_df.columns = ExtendedPatternBoosting.__tuples_to_strings(x_df.columns)

        choose_column_xgb_parameters = SettingsExtendedPatternBoosting().choose_column_xgb_parameters
        xgb_local_model = xgb.XGBRegressor(**choose_column_xgb_parameters)

        # y_target = pd.Series(np.random.randint(0, 9, y_target.shape))

        # x_df=pd.DataFrame(np.random.randint(0, 9, x_df.shape))

        xgb_local_model = xgb_local_model.fit(X=x_df, y=y_target)
        # xgb_local_model.get_booster().get_score(importance_type='weight')
        # xgb_local_model.get_booster().get_dump()

        # -------------------------------------------------------------------
        # delete me
        # we compute the error of the selector
        predictions = xgb_local_model.predict(x_df)
        mse = mean_squared_error(y_true=y_target, y_pred=predictions)
        print(f"selector {mse=}")
        # -------------------------------------------------------------------

        if SettingsExtendedPatternBoosting().show_tree is True:
            xgb.plot_tree(xgb_local_model)
            plt.show()

        selected_columns = np.argsort(xgb_local_model.feature_importances_)

        # xgb_local_model.get_booster().get_score(importance_type='gain')

        # alternative way to select best column using sklearn
        # xgb_local_model_tmp = xgb.XGBRegressor(**choose_column_xgb_parameters)
        # selector=SelectFromModel(xgb_local_model_tmp,threshold=-np.inf, max_features=1, prefit=False).fit(x_df, y_target)
        # best_path_tmp=selector.get_feature_names_out(x_df.columns)[0]

        selected_columns = selected_columns[-1]

        best_path = x_df.columns[selected_columns]
        del xgb_local_model

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
        if self.settings.name_model == 'xgboost':
            self.model = XgbModelForExtendedPathBoosting(**self.settings.main_xgb_parameters)
        elif self.settings.name_model == 'additive_xgboost':
            self.model = AdditiveXgbForExtendedPathBosting(**self.settings.main_xgb_parameters)
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
            if isinstance(train_boosting_matrix, pd.DataFrame):
                self.train_bm_df = train_boosting_matrix
            else:
                self.train_bm_df = pd.DataFrame(train_boosting_matrix, dtype='int')
                self.train_bm_df.columns = self.selected_paths
        elif isinstance(train_data, list):
            self.train_bm_df = ExtendedBoostingMatrix.create_boosting_matrix(selected_paths=self.selected_paths,
                                                                             list_graphs_nx=train_data,
                                                                             ebm_dataframe=self.train_ebm_dataframe)
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
            if isinstance(test_boosting_matrix,pd.DataFrame):
                self.test_bm_df = test_boosting_matrix
            else:
                self.test_bm_df = pd.DataFrame(test_boosting_matrix, dtype='int')
                self.test_bm_df.columns = self.selected_paths
        elif isinstance(test_data, list):
            self.test_bm_df = ExtendedBoostingMatrix.create_boosting_matrix(selected_paths=self.selected_paths,
                                                                            list_graphs_nx=test_data,
                                                                            ebm_dataframe=self.test_ebm_dataframe)
        else:
            raise Exception(
                "impossible to create boosting matrix for test data, provide boosting matrix or list of graph")

    def __training_results(self, X_test, y_test):

        predictions = self.model.predict(X_test)

        analysis.plot_error_evolution(error_list=self.model.test_error, dataset='test', show=True)
        analysis.plot_error_evolution(error_list=self.model.train_error, dataset='train', show=True)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_true=y_test, y_pred=predictions)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-Squared:", r2)
        print(f"{self.paths_selected_by_epb=}")

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
        return 2 * (y - y_hat)

    @staticmethod
    def __tuples_to_strings(tuples_list):
        return [str(tup) for tup in tuples_list]

    @staticmethod
    def __strings_to_tuples(strings_list):
        return [ast.literal_eval(s) for s in strings_list]
