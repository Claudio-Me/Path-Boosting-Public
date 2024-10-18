import copy
import os
import warnings

from sklearn import metrics
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from classes.boosting_matrix import BoostingMatrix
from classes.enumeration.model_type import ModelType
import numpy as np
from R_code.interface_with_R_code import LaunchRCode
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, root_mean_squared_error


class DecisionTreeModel:
    def __init__(self, settings: Settings):
        self.base_learners_list: list[tree.DecisionTreeRegressor] = []
        self.train_predictions_list: list[float] = []
        self.training_labels = None
        self.settings = settings
        self.parameters = settings.base_learner_tree_parameters

    def fit_one_step(self, boosting_matrix, labels):
        self.training_labels = labels
        if len(self.base_learners_list) == 0:
            # if it is the first time we launch the model
            # xgb_model = self.__create_xgb_model(base_score=None)
            base_tree = self.create_tree_base_learner()
            base_tree = base_tree.fit(boosting_matrix, labels)
            y_hat = base_tree.predict(boosting_matrix)
            self.train_predictions_list.append(y_hat)


        else:
            y_hat = self.train_predictions_list[-1]
            neg_gradient = self.__neg_gradient(y=labels, y_hat=y_hat, settings=self.settings)
            base_tree = self.create_tree_base_learner()
            base_tree = base_tree.fit(boosting_matrix, neg_gradient)
            new_y_hat = base_tree.predict(boosting_matrix)
            self.train_predictions_list.append(y_hat + new_y_hat)

        self.base_learners_list.append(base_tree)
        if self.settings.plot_tree is True:
            tree.plot_tree(self.base_learners_list[-1])
            plt.show()

        selected_column = np.argsort(self.base_learners_list[-1].feature_importances_)
        return selected_column[-1]

        # compute negative gradient
        # we can compute it just using the error of the last base learner

    def create_tree_base_learner(self) -> tree.DecisionTreeRegressor:
        tree_base_learner = tree.DecisionTreeRegressor(**self.parameters)
        return tree_base_learner

    def predict_my(self, boosting_matrix_matrix: np.ndarray | BoostingMatrix) -> np.ndarray:
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()

        predictions = []
        for base_learner in self.base_learners_list:
            single_prediction = base_learner.predict(boosting_matrix_matrix[:, 0:base_learner.n_features_in_])
            predictions.append(single_prediction)

        predictions = np.array(predictions)
        return predictions.sum(axis=0)

    def predict_progression(self, boosting_matrix_matrix: np.ndarray | BoostingMatrix,
                            from_training_data=False) -> np.array:
        # it returns the predictions at each step
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()
        if from_training_data is True:
            return self.train_predictions_list[-1]
        else:
            predictions = []
            for base_learner in self.base_learners_list:
                single_prediction = base_learner.predict(boosting_matrix_matrix[:, 0:base_learner.n_features_in_])
                predictions.append(single_prediction)

            predictions = np.array(predictions)
            return predictions.cumsum(axis=0)

    def evaluate_progression(self, boosting_matrix_matrix: np.ndarray | BoostingMatrix, labels,
                             evaluate_from_train_dataset=False) -> list[float]:
        # it returns the error at each step
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()
        y_pred = self.predict_progression(boosting_matrix_matrix, from_training_data=evaluate_from_train_dataset)
        error_list = []

        for prediction_with_i_base_learners in y_pred:
            if self.settings.final_evaluation_error == "MSE":
                error_list.append(metrics.mean_squared_error(labels, prediction_with_i_base_learners))
            elif self.settings.final_evaluation_error == "absolute_mean_error":
                error_list.append(metrics.mean_absolute_error(labels, prediction_with_i_base_learners))
            else:
                raise ValueError("measure error not found")
        return error_list

    def evaluate(self, boosting_matrix_matrix: np.ndarray | BoostingMatrix, labels):
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()

        y_pred = self.predict_my(boosting_matrix_matrix)
        y_pred = list(y_pred)
        if self.settings.final_evaluation_error == "MSE":
            model_error = metrics.mean_squared_error(labels, y_pred)
        elif self.settings.final_evaluation_error == "absolute_mean_error":
            model_error = metrics.mean_absolute_error(labels, y_pred)
        else:
            raise ValueError("measure error not found")
        return model_error

    def select_second_best_column(self, boosting_matrix: 'BoostingMatrix', first_column_number: int,
                                  labels: np.array, global_labels_variance: float | None = None) -> tuple[int, float]:
        """
        Selects the second-best column based on feature importance after fitting an XGBoost model on the boosting matrix
        without the first most important feature.

        Note, in case the boosting matrix has only one column, it will return first_column and variance of the labels as error

        This function is specifically designed for use with the XGBoost one-step model (xgb_one_step).
        It computes predictions and negative gradients for the current ensemble of models,
        excluding the last model. It creates a new BoostingMatrix without the specified column
        (first_column_number) and fits an XGBoost regression model to the negative gradients.
        The function then selects the column with the second-highest feature importance based on this fitted model.

        :param global_labels_variance:
        :type global_labels_variance: float | None
        :param boosting_matrix: An instance of BoostingMatrix containing the data matrix.
        :type boosting_matrix: BoostingMatrix
        :param first_column_number: Index of the column to be excluded from the BoostingMatrix when fitting the XGBoost model.
        :type first_column_number: int
        :param labels: The array of true label values used to compute the negative gradient.
        :type labels: np.array
        :raises Warning: If the model type is not ModelType.xgb_one_step, a warning is raised.
        :return: A tuple with the index of the second-best column (with the second-highest feature importance) and the final training error of the model.
        :rtype: tuple[int, float]


        """
        if isinstance(boosting_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix.get_matrix()
        else:
            boosting_matrix_matrix = boosting_matrix

        predictions = []
        for base_learner in self.base_learners_list[:-1]:
            single_prediction = base_learner.predict(boosting_matrix_matrix[:, 0:base_learner.n_features_in_])
            predictions.append(single_prediction)

        predictions = np.array(predictions)

        y_hat = predictions.sum(axis=0)

        neg_gradient = self.__neg_gradient(y=labels, y_hat=y_hat, settings=self.settings)

        if len(boosting_matrix.get_header()) <= 1:
            if global_labels_variance is None:
                final_train_error = np.var(labels)
                return first_column_number, final_train_error

            else:
                return first_column_number, global_labels_variance

        # remove the selected column from the matrix and train a model over the new matrix
        boosting_matrix_without_column = boosting_matrix.new_matrix_without_column(first_column_number)
        boosting_matrix_without_column_matrix = boosting_matrix_without_column.get_matrix()

        tree_base_learner = self.create_tree_base_learner()

        tree_base_learner = tree_base_learner.fit(X=boosting_matrix_without_column_matrix, y=neg_gradient)
        selected_column = np.argsort(tree_base_learner.feature_importances_)
        selected_column = selected_column[-1]
        if selected_column >= first_column_number:
            # this is because second selected column will be actually one spot after the addition of selected column
            selected_column += 1

        predictions = tree_base_learner.predict(boosting_matrix_without_column_matrix)

        # compute mse
        final_train_error = mean_squared_error(y_true=neg_gradient, y_pred=predictions)

        return selected_column, final_train_error

    @staticmethod
    def __neg_gradient(y, y_hat, settings):
        return settings.neg_gradient(y=y, y_hat=y_hat)

    def get_last_training_error(self) -> float:
        # expected to return rmse
        predictions = self.train_predictions_list[-1]
        last_train_error = mean_squared_error(y_true=self.training_labels, y_pred=predictions)
        return last_train_error
