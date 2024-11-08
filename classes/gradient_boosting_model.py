import copy
import os
import warnings

import xgboost
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics


from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from classes.boosting_matrix import BoostingMatrix
from classes.enumeration.model_type import ModelType
import numpy as np
from R_code.interface_with_R_code import LaunchRCode
from xgboost import plot_tree
import matplotlib.pyplot as plt


class GradientBoostingModel:
    def __init__(self, model, settings : Settings = None):

        # note: this two 'if' are useless since they are doing the same operation, I leave them there just in case I want to modify the code later
        if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
            self.model = model
            self.n_features = None
        elif model is ModelType.r_model:
            # function that call R script, it will be allocated later
            self.r_select_column_and_train_model = None

            self.model = model
        elif model is ModelType.xgb_one_step:
            self.base_learners_list = []
            self.base_learners_dimension = []
            self.model = model
        if settings is None:
            self. settings = Settings()
        else:
            self.settings = settings

    def predict_my(self, boosting_matrix_matrix: np.ndarray | BoostingMatrix):
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()

        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.predict(boosting_matrix_matrix[:, :self.n_features])
        elif self.model is ModelType.r_model:
            r_predict_model = LaunchRCode(self.settings.r_code_relative_location, "main_predict")
            predictions_vector = r_predict_model.r_function(np.array(boosting_matrix_matrix), self.settings.r_model_name,
                                                            os.path.join(os.getcwd(), "R_code"))
            del r_predict_model
            return predictions_vector

        elif self.model is ModelType.xgb_one_step:
            """
            predictions = np.array([xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]) for
                                    xgb_model, matrix_dimension in
                                    zip(self.base_learners_list, self.base_learners_dimension)])
            """
            predictions = []
            for xgb_model, matrix_dimension in zip(self.base_learners_list, self.base_learners_dimension):
                predictions.append(xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]))
            predictions=np.array(predictions)

            return predictions.sum(axis=0)

    def predict_progression(self, boosting_matrix_matrix: np.ndarray):
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()

        if self.model is not ModelType.xgb_one_step:
            raise ValueError("predict_progression only works with 'Xgb_step' algorithm")
        else:
            """
            predictions = np.array([xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]) for
                                    xgb_model, matrix_dimension in
                                    zip(self.base_learners_list, self.base_learners_dimension)])
            """
            predictions = []
            for xgb_model, matrix_dimension in zip(self.base_learners_list, self.base_learners_dimension):
                predictions.append(xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]))
            predictions=np.array(predictions)
            return predictions.cumsum(axis=0)

    def evaluate_progression(self, boosting_matrix_matrix, labels) -> list[float]:
        if isinstance(boosting_matrix_matrix, BoostingMatrix):
            boosting_matrix_matrix = boosting_matrix_matrix.get_matrix()
        if self.model is not ModelType.xgb_one_step:
            raise ValueError("evaluate_progression only works with 'Xgb_step' algorithm")
        else:
            y_pred = self.predict_progression(boosting_matrix_matrix)
            y_pred = list(y_pred)

            error_list = []

            for prediction_with_i_base_learners in y_pred:
                if self.settings.final_evaluation_error == "MSE":
                    error_list.append(metrics.mean_squared_error(labels, prediction_with_i_base_learners))
                elif self.settings.final_evaluation_error == "absolute_mean_error":
                    error_list.append(metrics.mean_absolute_error(labels, prediction_with_i_base_learners))
                else:
                    raise ValueError("measure error not found")
            return error_list

    def evaluate(self, boosting_matrix_matrix, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            y_pred = self.predict_my(boosting_matrix_matrix)

            if self.settings.estimation_type is EstimationType.classification:
                y_pred = [round(value) for value in y_pred]

        elif self.model is ModelType.r_model:
            y_pred = list(self.predict_my(boosting_matrix_matrix))

        elif self.model is ModelType.xgb_one_step:
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

        .. warning:: This function is only applicable for ModelType.xgb_one_step. If called with a different model type, it raises a warning.
        """

        boosting_matrix_matrix = boosting_matrix.get_matrix()
        if self.model is ModelType.xgb_one_step:

            predictions = np.array([xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]) for
                                    xgb_model, matrix_dimension in
                                    zip(self.base_learners_list[:-1], self.base_learners_dimension[:-1])])
            y_hat = predictions.sum(axis=0)

            neg_gradient = self.__neg_gradient(labels, y_hat)

            if len(boosting_matrix.get_header()) <= 1:
                if global_labels_variance is None:
                    final_train_error = np.var(labels)
                    return first_column_number, final_train_error

                else:
                    return first_column_number, global_labels_variance

            # remove the selected column from the matrix and train a model over the new matrix
            boosting_matrix_without_column = boosting_matrix.new_matrix_without_column(first_column_number)
            boosting_matrix_without_column_matrix = boosting_matrix_without_column.get_matrix()

            xgb_model = self.__create_xgb_model(base_score=np.mean(neg_gradient))

            eval_set = [(boosting_matrix_without_column_matrix, neg_gradient)]
            xgb_model.fit(X=boosting_matrix_without_column_matrix, y=neg_gradient, eval_set=eval_set)
            selected_column = np.argsort(xgb_model.feature_importances_)
            selected_column = selected_column[-1]
            if selected_column >= first_column_number:
                # this is because secon selected column will be actually one spot after the addition of selected column
                selected_column += 1

            results = xgb_model.evals_result()
            final_train_error = results['validation_0'][self.settings.xgb_model_parameters["eval_metric"]][-1]

            # we assume the eval method is rmse
            final_train_error = final_train_error * final_train_error
            return selected_column, final_train_error

        else:
            warnings.warn("Selection of second best column is possible only with model type xgb_one_step")

    def fit_one_step(self, boosting_matrix: np.ndarray, labels):
        # ----------------------------------------------------------------------------------------------------------

        # N.B. this function returns the selected column and trains the model

        # ----------------------------------------------------------------------------------------------------------
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):

            self.model.fit(boosting_matrix, labels)
            self.n_features = len(boosting_matrix[0])
            return np.argsort(self.model.feature_importances_)[0]

        elif self.model is ModelType.r_model:
            # if self.r_select_column_and_train_model is None:
            #   self.r_select_column_and_train_model = LaunchRCode(self.settings.r_code_relative_location, "select_column")

            self.r_select_column_and_train_model = LaunchRCode(self.settings.r_code_relative_location, "select_column")

            selected_column_number = self.r_select_column_and_train_model.r_function(np.array(boosting_matrix),
                                                                                     np.array(labels),
                                                                                     self.settings.r_model_name,
                                                                                     os.path.join(os.getcwd(),
                                                                                                  "R_code"),
                                                                                     self.settings.family,
                                                                                     self.settings.r_base_learner_name)
            del self.r_select_column_and_train_model
            return selected_column_number

        elif self.model is ModelType.xgb_one_step:
            if len(self.base_learners_list) == 0:
                # if it is the first time we launch the model
                # xgb_model = self.__create_xgb_model(base_score=None)
                xgb_model = self.__create_xgb_model(base_score=np.mean(labels))

                eval_set = [(boosting_matrix, labels)]
                xgb_model.fit(boosting_matrix, labels, eval_set=eval_set)

                # plot single tree
                if self.settings.plot_tree is True:
                    plot_tree(xgb_model)
                    plt.show()

                self.base_learners_list.append(copy.deepcopy(xgb_model))
                self.base_learners_dimension.append(len(boosting_matrix[0]))
                selected_column = np.argsort(xgb_model.feature_importances_)
                return selected_column[-1]
            else:

                y_hat = self.predict_my(boosting_matrix)

                # ----------------------------------------------------------------------------------------------------
                # compute the residuals, they should coincide with the residuals of the last model

                # ----------------------------------------------------------------------------------------------------

                neg_gradient = self.__neg_gradient(labels, y_hat)

                xgb_model = self.__create_xgb_model(base_score=np.mean(neg_gradient))
                eval_set = [(boosting_matrix, neg_gradient)]
                xgb_model.fit(X=boosting_matrix, y=neg_gradient, eval_set=eval_set, verbose=self.settings.verbose)

                # plot single tree
                if self.settings.plot_tree is True:
                    plot_tree(xgb_model)
                    plt.show()

                self.base_learners_list.append(copy.deepcopy(xgb_model))
                # --------------------------------------------------------------------------------------------------
                '''
                print('Access logloss metric directly from evals_result:')
                print(evals_result['eval']['logloss'])

                print('')
                print('Access metrics through a loop:')
                for e_name, e_mtrs in evals_result.items():
                    print('- {}'.format(e_name))
                    for e_mtr_name, e_mtr_vals in e_mtrs.items():
                        print('   - {}'.format(e_mtr_name))
                        print('      - {}'.format(e_mtr_vals))

                print('')
                print('Access complete dictionary:')
                print(evals_result)
                '''
                # plot xgb model
                # plot_tree(xgb_model, num_trees=0)
                # plt.show()
                # --------------------------------------------------------------------------------------------------
                self.base_learners_dimension.append(len(boosting_matrix[0]))
                selected_column = np.argsort(xgb_model.feature_importances_)

                # --------------------------------------------------------------------------------------------------
                # debug
                # look at the error of the last base model
                # tmp_new_predicted_y = xgb_model.predict(boosting_matrix)

                # model_error = metrics.mean_squared_error(tmp_new_predicted_y, neg_gradient)
                # model_error = np.sqrt(model_error)
                # print("Base learner rmse: ", model_error)

                # -------------------------------------------------------------------------------------------------

                return selected_column[-1]

    def __create_xgb_model(self, base_score=0.0):

        # create a Xgb model
        param = self.settings.xgb_model_parameters
        if self.settings.estimation_type is EstimationType.regression:

            return XGBRegressor(**self.settings.xgb_model_parameters,
                                base_score=base_score)
        elif self.settings.estimation_type is EstimationType.classification:
            return XGBClassifier(param, num_boosted_rounds=2)
        else:
            TypeError("Estimation task not recognized")

    def __neg_gradient(self, y, y_hat):
        return self.settings.neg_gradient(y, y_hat)

    def get_last_training_error(self):
        if self.model is ModelType.xgb_one_step:
            last_base_learner = self.base_learners_list[-1]

            results = last_base_learner.evals_result()
            return results['validation_0'][self.settings.xgb_model_parameters['eval_metric']][-1]
        else:
            raise TypeError("Can't get last training error for this model")
