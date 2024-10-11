import matplotlib.pyplot as plt
import pandas as pd
from fontTools.varLib.errors import NotANone
import warnings
from classes.extended_boosting_matrix import ExtendedBoostingMatrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
import numpy as np

class AdditiveTreeModelForExtendedPathBosting:
    def __init__(self, **kwargs):
        # we expect each model that is passed to have the following methods:
        # .fit
        # .predict
        # .refit
        # .get_model
        # .get_dump
        # .plot_tree

        self.__kwargs = kwargs
        self.__last_train_prediction: pd.Series | None = None
        self.__last_test_prediction: pd.Series | None = None
        self.train_error = []
        self.test_error = []
        self.base_learners_list: list[DecisionTreeRegressor] = []

    def fit(self, X, y, best_path, dict_of_interaction_constraints, eval_set=None, negative_gradient=None):
        # we assume y is the real target and we compute the negative gradient
        # note: in case eval_set is passed, we assume the first parameter is the training dataset.
        # create new base learner model

        zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
            x_df=X, y=y, path=best_path, dict_of_interaction_constraints=dict_of_interaction_constraints)

        # done because DecisionTreeRegressor doe not handle nan values
        zeroed_x_df.replace(np.nan,-10, inplace=True)

        new_base_learner = DecisionTreeRegressor(**self.__kwargs)

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient
            new_base_learner.fit(zeroed_x_df, zeroed_y)
            self.base_learners_list.append(new_base_learner)

            self.__last_train_prediction = pd.Series(new_base_learner.predict(X))
            train_mse = mean_squared_error(y_true=y, y_pred=self.__last_train_prediction)
            self.train_error.append(train_mse)

            if eval_set is not None:
                # we assume eval_set[1][0] is the test dataset
                self.__last_test_prediction = pd.Series(new_base_learner.predict(eval_set[1][0]))
                test_mse = mean_squared_error(y_true=eval_set[1][1], y_pred=self.__last_test_prediction)
                self.test_error.append(test_mse)




        else:

            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)
            if negative_gradient is None:
                negative_gradient = self.__neg_gradient(y=y, y_hat=self.__last_train_prediction)
            new_y = pd.Series(negative_gradient).loc[zeroed_y.index]

            new_base_learner = DecisionTreeRegressor(**self.__kwargs)
            new_base_learner.fit(zeroed_x_df, new_y)

            self.base_learners_list.append(new_base_learner)

            self.__last_train_prediction += new_base_learner.predict(X)
            train_mse = mean_squared_error(y_true=y, y_pred=self.__last_train_prediction)
            self.train_error.append(train_mse)

            if eval_set is not None:
                # we assume eval_set[1][0] is the test dataset
                self.__last_test_prediction += new_base_learner.predict(eval_set[1][0])
                test_mse = mean_squared_error(y_true=eval_set[1][1], y_pred=self.__last_test_prediction)
                self.test_error.append(test_mse)

        return self

    def predict(self, X, **kwargs):
        prediction = []
        for base_learner in self.base_learners_list:
            prediction.append(base_learner.predict(X, **kwargs))

        return sum(prediction)

    def get_dump(self):
        warnings.warn("Get dump is note implemented for additive tree model")
        return None

    def get_model(self):
        return self.base_learners_list

    def plot_tree(self):
        for base_learner in self.base_learners_list:
            tree.plot_tree(base_learner)
            plt.show()

    @staticmethod
    def __neg_gradient(y, y_hat):
        return y - y_hat
