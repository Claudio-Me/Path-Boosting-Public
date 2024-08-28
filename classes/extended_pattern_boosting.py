import xgboost as xgb
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


class ExtendedPatternBoosting(ExtendedBoostingMatrix):

    def __init__(self, extended_boosting_matrix: ExtendedBoostingMatrix | None = None,
                 nx_graphs_dataset: list[nx.classes.multigraph.MultiGraph] = None, boosting_matrix=None,
                 selected_paths: list[tuple[(int)]] | None = None,
                 settings: SettingsExtendedPatternBoosting | None = None,
                 dict_of_interaction_constraints: dict | None = None):

        super().__init__()

        # they are initialized in the function 'initialize_expanded_boosting_matrix'
        self.dict_of_interaction_constraints: dict | None = None
        self.test_ebm_dataframe: pd.DataFrame | None = None
        self.extended_boosting_matrix: ExtendedBoostingMatrix | None = None
        self.selected_paths: list[tuple[(int)]] | None = None
        self.train_ebm_dataframe: pd.DataFrame | None = None
        if settings is None:
            self.settings: SettingsExtendedPatternBoosting = SettingsExtendedPatternBoosting()
        else:
            self.settings: SettingsExtendedPatternBoosting = settings

        all_are_none = all(variable is None for variable in [
            extended_boosting_matrix,
            nx_graphs_dataset,
            boosting_matrix,
            selected_paths,
            settings,
            dict_of_interaction_constraints
        ])
        if all_are_none is False:
            self.initialize_expanded_pattern_boosting(selected_paths=selected_paths,
                                                      extended_boosting_matrix=extended_boosting_matrix,
                                                      list_graphs_nx=nx_graphs_dataset,
                                                      boosting_matrix=boosting_matrix,
                                                      dict_of_interaction_constraints=dict_of_interaction_constraints)

    def train(self, train_data: pd.DataFrame | None = None, dict_of_interaction_constraints: dict | None = None,
              test_data=None,
              extended_boosting_matrix: ExtendedBoostingMatrix | None = None,
              list_graphs_nx: list[nx.classes.multigraph.MultiGraph] = None,
              boosting_matrix: BoostingMatrix | None = None,
              selected_paths: list[tuple[int]] | None = None):

        assert isinstance(train_data, pd.DataFrame) or train_data is None
        assert isinstance(dict_of_interaction_constraints, dict) or dict_of_interaction_constraints is None

        if (train_data is not None) and (dict_of_interaction_constraints is None):
            raise ValueError("dict_of_interaction_constraints should both be given in input, got ",
                             dict_of_interaction_constraints)
        if train_data is not None:
            self.train_ebm_dataframe: pd.DataFrame = train_data
        if self.train_ebm_dataframe is None:
            self.initialize_expanded_pattern_boosting(selected_paths, extended_boosting_matrix, list_graphs_nx,
                                                      boosting_matrix, dict_of_interaction_constraints)

        self.set_up_test_data(test_data)

        # train, test = train_test_split(self.extended_boosting_matrix_dataframe, test_size=0.2, random_state=0)



        #self.settings.xgb_parameters["interaction_constraints"]=self.dict_of_interaction_constraints.items
        train_target = self.train_ebm_dataframe['target']

        train = self.train_ebm_dataframe.drop(['target'], axis=1)

        test_target = self.test_ebm_dataframe['target']

        test = self.test_ebm_dataframe.drop(['target'], axis=1)

        #train = self.train_ebm_dataframe['target']
        #test = self.test_ebm_dataframe['target']


        evallist = [(test, test_target)]



        # xgb_model = xgb.XGBRegressor(**parameters)

        xgb_model = xgb.XGBRegressor(**self.settings.xgb_parameters)
        xgb_model = xgb_model.fit( train, train_target, eval_set=evallist)



        if self.settings.plot_analysis is True:
            ExtendedPatternBoosting.training_results(bst=xgb_model, X_test=test, y_test=test_target)

    def initialize_expanded_pattern_boosting(self, selected_paths: list[tuple[int]] | None = None,
                                             extended_boosting_matrix: ExtendedBoostingMatrix | None = None,
                                             list_graphs_nx: list[nx.classes.multigraph.MultiGraph] = None,
                                             boosting_matrix: BoostingMatrix | None = None,
                                             dict_of_interaction_constraints: dict | None = None):

        # self.nx_graphs_dataset = nx_graphs_dataset
        # self.boosting_matrix = boosting_matrix
        # self.selected_paths = selected_paths

        if hasattr(selected_paths, '__iter__'):
            self.selected_paths = selected_paths
        elif isinstance(boosting_matrix, BoostingMatrix):
            self.selected_paths = boosting_matrix.get_selected_paths()

        if extended_boosting_matrix is not None:
            self.train_ebm_dataframe = extended_boosting_matrix.get_pandas_dataframe()
        elif list_graphs_nx is not None:
            self.train_ebm_dataframe = ExtendedBoostingMatrix.create_extend_boosting_matrix_for(
                selected_paths=selected_paths, list_graphs_nx=list_graphs_nx)

        if dict_of_interaction_constraints is None:
            self.dict_of_interaction_constraints = ExtendedBoostingMatrix.get_features_interaction_constraints(
                list_graphs_nx=list_graphs_nx, selected_paths=self.selected_paths)
        else:
            self.dict_of_interaction_constraints = dict_of_interaction_constraints

    def set_up_test_data(self, test_data):
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
        my_r2 =calculate_r2_score(y_true=y_test, y_pred=predictions)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-Squared:", r2)
        print("My R-Squared:", my_r2)

        # Plotting the feature importance
        #xgb.plot_importance(bst)
        #plt.show()

        # Scatter plot of actual vs predicted values
        # modify the color map
        cmap = mpl.cm.Blues(np.linspace(0, 1, 100))
        cmap = mpl.colors.ListedColormap(cmap[20:, :-1])

        # Create a new figure and a set of subplots
        fig, ax = plt.subplots()

        # Hexbin plot with adjusted vmin parameter
        # We set vmin to a fraction, this will make a single point darker than without setting vmin
        hb = ax.hexbin(y_test, predictions, gridsize=50, cmap=cmap, mincnt=1)
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label('Density')

        # Chart details
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')

        # Identity line
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)

        # Display plot
        plt.show()


        # Optional: Return the metrics if you need them for further analysis
        return {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }