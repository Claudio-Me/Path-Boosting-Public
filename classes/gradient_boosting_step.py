from typing import List, Tuple, Any
from classes.enumeration.estimation_type import EstimationType
from classes.gradient_boosting_model import GradientBoostingModel
from sklearn import metrics
from settings import Settings
from classes.boosting_matrix import BoostingMatrix

if Settings().algorithm == "R":
    from R_code.interface_with_R_code import LaunchRCode
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from collections.abc import Iterable
import gc
import os
import numpy as np
from classes.enumeration.model_type import ModelType
from classes.decision_tree_model import DecisionTreeModel


class GradientBoostingStep:
    def __init__(self, settings: Settings) -> None:
        self.training_error: list[float] = []
        self.number_of_learners: list[int] = []
        if settings is None:
            self.settings = Settings()
        else:
            self.settings = settings

    def select_column(self, model, boosting_matrix: BoostingMatrix, labels: list, number_of_learners: int = None):
        """It makes one step of gradient boosting ant it returns the selected column, with the trained model"""
        if self.settings.algorithm == "R":

            return self.__step_using_r(model, boosting_matrix, labels)

        elif self.settings.algorithm == "Full_xgb":
            return self.__step_training_a_whole_new_XGB_model(boosting_matrix, labels, number_of_learners)

        elif self.settings.algorithm == "Xgb_step":

            return self.__step_using_xgboost(model, boosting_matrix, labels)
        elif self.settings.algorithm == "decision_tree":
            return self.__step_using_decision_trees(model, boosting_matrix, labels)
        else:
            raise TypeError("Selected algorithm not recognized")

    def __step_using_decision_trees(self, model, boosting_matrix, labels):
        if model is None:
            model = DecisionTreeModel(settings=self.settings)
        selected_column = model.fit_one_step(boosting_matrix.matrix, labels)
        # -------------------------------------------------------------------------------------------------------------
        if self.settings.verbose is True:
            print("Selected column ", selected_column)
        # -------------------------------------------------------------------------------------------------------------
        return selected_column, model

    def __step_using_xgboost(self, model: GradientBoostingModel, boosting_matrix: BoostingMatrix, labels):
        if model is None:
            model = GradientBoostingModel(model=ModelType.xgb_one_step, settings=self.settings)
        selected_column = model.fit_one_step(boosting_matrix.matrix, labels)
        # -------------------------------------------------------------------------------------------------------------
        if self.settings.verbose is True:
            print("Selected column ", selected_column)
        # -------------------------------------------------------------------------------------------------------------
        return selected_column, model

    def __step_using_r(self, model, boosting_matrix: BoostingMatrix, labels):
        if model is None:
            # we are at the first step, the model is no initialized yet
            r_select_column_and_train_model = LaunchRCode(self.settings.r_code_relative_location, "first_iteration")
            selected_column_number = r_select_column_and_train_model.r_function(np.array(boosting_matrix.matrix),
                                                                                np.array(labels),
                                                                                self.settings.r_model_name,
                                                                                os.path.join(os.getcwd(), "R_code"),
                                                                                self.settings.family,
                                                                                self.settings.r_base_learner_name)
            model = GradientBoostingModel(model=ModelType.r_model, settings=self.settings)
            del r_select_column_and_train_model
        else:
            selected_column_number = model.fit_one_step(np.array(boosting_matrix.matrix), np.array(labels))

        if isinstance(selected_column_number, Iterable):
            selected_column_number = selected_column_number[0]
        selected_column_number = int(selected_column_number)
        gc.collect()
        return selected_column_number, model

    def __step_training_a_whole_new_XGB_model(self, boosting_matrix: BoostingMatrix, labels: list,
                                              number_of_learners: int):

        if self.settings.estimation_type is EstimationType.regression:
            xgb_model = XGBRegressor(n_estimators=number_of_learners)
        elif self.settings.estimation_type is EstimationType.classification:
            xgb_model = XGBClassifier(n_estimators=number_of_learners)
        else:
            TypeError("Estimation task not recognized")

        xgb_model = GradientBoostingModel(model=xgb_model, settings=self.settings)
        xgb_model.fit_one_step(boosting_matrix.matrix, labels)

        """
        y_pred = xgb_model.predict(boosting_matrix.matrix)

        
        if self.settings.estimation_type is EstimationType.regression:
            error = metrics.mean_squared_error(labels, y_pred)
        elif self.settings.estimation_type is EstimationType.classification:
            y_pred = [round(value) for value in y_pred]
            error = metrics.accuracy_score(labels, y_pred)
        """

        error = xgb_model.evaluate(boosting_matrix.matrix, labels)
        self.training_error.append(error)
        self.number_of_learners.append(number_of_learners)

        features_order = np.argsort(xgb_model.model.feature_importances_)
        for selected_feature_column in features_order:
            if not (selected_feature_column in boosting_matrix.already_selected_columns):
                if not isinstance(int(selected_feature_column), int):
                    raise TypeError("Debug: selected column is wrong")
                return int(selected_feature_column), xgb_model

        raise TypeError("Impossible to find a novel column to expand")
