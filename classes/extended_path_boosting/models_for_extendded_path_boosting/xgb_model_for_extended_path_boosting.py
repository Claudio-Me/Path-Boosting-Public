import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classes.extended_boosting_matrix import ExtendedBoostingMatrix


class XgbModelForExtendedPathBoosting:
    def __init__(self, **xgb_parameters):
        # we expect each model that is passed to have the following methods:
        # .fit
        # .predict
        # .refit
        # .get_model
        # .get_dump
        # .plot_tree

        self.train_error = []
        self.test_error = []

        if xgb_parameters["objective"] is None:
            xgb_parameters["objective"] = self.custom_objective_function

        self.model: xgb.XGBRegressor = xgb.XGBRegressor(**xgb_parameters)
        self.__fitted = False

    def fit(self, X, y, best_path, dict_of_interaction_constraints, eval_set=None, negative_gradient=None):
        # negative gradient is supposed to be the negative gradient computed on the original dataset
        # note: we assume the first parameter in eval_set is the training dataset and eval_set[0][1] are the original labels.

        zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
            x_df=X, y=y, path=best_path, dict_of_interaction_constraints=dict_of_interaction_constraints)

        if self.__fitted is False:
            self.model.fit(zeroed_x_df, zeroed_y, eval_set=eval_set)
            self.__fitted = True
        else:
            # we have to compute the negative gradient on the training matrix because otherwise the model will train on the neg gradient from the zeroed matrix
            if negative_gradient is None:
                negative_gradient = self.__neg_gradient(y=y, y_hat=self.predict(X))

            negative_gradient_for_zeroed_matrix = pd.Series(negative_gradient).loc[zeroed_y.index]
            y_zero_hat = self.model.predict(zeroed_x_df)
            new_target = y_zero_hat + negative_gradient_for_zeroed_matrix

            self.model.fit(zeroed_x_df, new_target, eval_set=eval_set, xgb_model=self.model)

        self.train_error.append(pow(self.model.evals_result()['validation_0']['rmse'][-1], 2))
        try:
            self.test_error.append(pow(self.model.evals_result()['validation_1']['rmse'][-1], 2))
        except:
            pass
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def get_dump(self):
        return self.model.get_booster().get_dump()

    def get_model(self):
        return self.model

    def plot_tree(self, num_trees: int = -1):
        xgb.plot_tree(self.model.get_booster(), num_trees=num_trees)
        plt.show()

    @staticmethod
    def custom_objective_function(y_true, y_pred):

        gradient = y_true  # Difference between predicted and true values
        hessian = np.ones_like(gradient)  # Array of ones, same shape as gradient
        return gradient, hessian

    @staticmethod
    def __neg_gradient(y, y_hat):
        return 2 * (y - y_hat)
