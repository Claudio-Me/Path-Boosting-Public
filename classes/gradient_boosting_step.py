from typing import List
from classes.enumeration.estimation_type import EstimationType

from sklearn import metrics
from settings import Settings
from classes.boosting_matrix import BoostingMatrix
from R_code.interface_with_R_code import LaunchRCode
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import random


class GradientBoostingStep:
    def __init__(self):
        self.training_accuracy: list[float] = []
        self.number_of_learners: list[int] = []

    def select_column(self, boosting_matrix: BoostingMatrix, labels: list, number_of_learners: int = None) -> int:
        """It makes one step of gradient boosting ant it returns the selected column"""
        if Settings.use_R is True:
            return self.__step_using_r(boosting_matrix, labels)

        else:
            return self.__step_using_python(boosting_matrix, labels, number_of_learners)

    def __step_using_r(self, boosting_matrix: BoostingMatrix, labels) -> int:
        r_interface = LaunchRCode()
        selected_column_number = r_interface.launch_function(boosting_matrix.matrix, labels)
        return selected_column_number

    def __step_using_python(self, boosting_matrix: BoostingMatrix, labels: list, number_of_learners: int) -> int:

        if Settings.estimation_type is EstimationType.regression:
            xgb_model = XGBRegressor(n_estimators=number_of_learners)
        elif Settings.estimation_type is EstimationType.classification:
            xgb_model = XGBClassifier(n_estimators=number_of_learners)
        else:
            TypeError("Estimation task not recognized")

        xgb_model.fit(boosting_matrix.matrix, labels)

        y_pred = xgb_model.predict(boosting_matrix.matrix)

        if Settings.estimation_type is EstimationType.regression:
            accuracy = metrics.mean_squared_error(labels, y_pred)
        elif Settings.estimation_type is EstimationType.classification:
            y_pred = [round(value) for value in y_pred]
            accuracy = metrics.accuracy_score(labels, y_pred)

        self.training_accuracy.append(accuracy)
        self.number_of_learners.append(number_of_learners)

        features_order = np.argsort(xgb_model.feature_importances_)
        for selected_feature_column in features_order:
            if not (selected_feature_column in boosting_matrix.already_selected_columns):
                return selected_feature_column

        raise TypeError("Impossible to find a novel column to expand")

    def plot_training_accuracy(self):

        fig, ax = plt.subplots()

        # Using set_dashes() to modify dashing of an existing line
        ax.plot(self.number_of_learners, self.training_accuracy, label='Training accuracy')
        ax.set_title('Training accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Number of learners')

        # plot only integers on the x axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.show()
