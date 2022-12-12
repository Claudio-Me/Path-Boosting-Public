from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from classes.enumeration.model_type import ModelType
import numpy as np
from R_code.interface_with_R_code import LaunchRCode
from xgboost import plot_tree
import matplotlib.pyplot as plt


class GradientBoostingModel:
    def __init__(self, model):

        # note: this two 'if' are useless since they are doing the same operation, I leave them there just in case I want to modify the code later
        if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
            self.model = model
        elif model is ModelType.r_model:
            # function that call R script, it will be allocated later
            self.r_select_column_and_train_model = None

            self.model = model
        elif model is ModelType.xgb_one_step:
            self.base_learners_list = []
            self.base_learners_dimension = []
            self.model = model

    def predict_my(self, boosting_matrix_matrix: np.ndarray):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.predict(boosting_matrix_matrix)
        elif self.model is ModelType.r_model:
            r_predict_model = LaunchRCode(Settings.r_code_relative_location, "main_predict")
            predictions_vector = r_predict_model.r_function(np.array(boosting_matrix_matrix), Settings.r_model_name,
                                                            Settings.r_model_location)
            del r_predict_model
            return predictions_vector

        elif self.model is ModelType.xgb_one_step:
            predictions = np.array([xgb_model.predict(boosting_matrix_matrix[:, 0:matrix_dimension]) for
                                    xgb_model, matrix_dimension in
                                    zip(self.base_learners_list, self.base_learners_dimension)])

            return predictions.sum(axis=0)

    def evaluate(self, boosting_matrix_matrix, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            y_pred = self.predict_my(boosting_matrix_matrix)

            if Settings.estimation_type is EstimationType.classification:
                y_pred = [round(value) for value in y_pred]

        elif self.model is ModelType.r_model:
            y_pred = list(self.predict_my(boosting_matrix_matrix))

        elif self.model is ModelType.xgb_one_step:
            y_pred = self.predict_my(boosting_matrix_matrix)
            y_pred = list(y_pred)

        if Settings.final_evaluation_error == "MSE":
            model_error = metrics.mean_squared_error(labels, y_pred)
        elif Settings.final_evaluation_error == "absolute_mean_error":
            model_error = metrics.mean_absolute_error(labels, y_pred)
        else:
            raise ValueError("measure error not found")
        return model_error

    def fit_one_step(self, boosting_matrix: np.ndarray, labels):
        # ----------------------------------------------------------------------------------------------------------

        # N.B. this function returns the selected column and trains the model

        # ----------------------------------------------------------------------------------------------------------
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):

            self.model.fit(boosting_matrix, labels)

            return np.argsort(self.model.feature_importances_)[0]

        elif self.model is ModelType.r_model:
            # if self.r_select_column_and_train_model is None:
            #   self.r_select_column_and_train_model = LaunchRCode(Settings.r_code_relative_location, "select_column")

            self.r_select_column_and_train_model = LaunchRCode(Settings.r_code_relative_location, "select_column")

            selected_column_number = self.r_select_column_and_train_model.r_function(np.array(boosting_matrix),
                                                                                     np.array(labels),
                                                                                     Settings.r_model_name,
                                                                                     Settings.r_model_location,
                                                                                     Settings.family,
                                                                                     Settings.r_base_learner_name)
            del self.r_select_column_and_train_model
            return selected_column_number

        elif self.model is ModelType.xgb_one_step:
            if len(self.base_learners_list) == 0:
                # if it is the first time we launch the model
                # xgb_model = self.__create_xgb_model(np.mean(labels))
                xgb_model = self.__create_xgb_model(0)
                xgb_model.fit(boosting_matrix, labels)
                self.base_learners_list.append(xgb_model)
                self.base_learners_dimension.append(len(boosting_matrix[0]))
                selected_column = np.argsort(xgb_model.feature_importances_)
                return selected_column[-1]
            else:

                y_hat = self.predict_my(boosting_matrix)

                # ----------------------------------------------------------------------------------------------------
                # compute the residuals, they should coincide with the residuals of the last model

                # ----------------------------------------------------------------------------------------------------

                neg_gradient = self.__neg_gradient(labels, y_hat)
                # xgb_model = self.__create_xgb_model(np.mean(neg_gradient))
                xgb_model = self.__create_xgb_model(base_score=np.mean(neg_gradient),
                                                    estimation_type=EstimationType.regression)

                bst = xgb_model.fit(boosting_matrix, neg_gradient)

                self.base_learners_list.append(xgb_model)
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
                return selected_column[-1]

    def __create_xgb_model(self, base_score=0, estimation_type=Settings.estimation_type):
        # create a Xgb model
        if estimation_type is EstimationType.regression:
            return XGBRegressor(n_estimators=1, max_depth=1, booster="gbtree", base_score=base_score, learning_rate=0.1)
        elif estimation_type is EstimationType.classification:
            return XGBClassifier(num_boosted_rounds=2)
        else:
            TypeError("Estimation task not recognized")

    def __neg_gradient(self, y, y_hat):
        return Settings.neg_gradient(y, y_hat)
