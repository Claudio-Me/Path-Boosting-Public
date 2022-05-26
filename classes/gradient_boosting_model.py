from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
import numpy as np


class GradientBoostingModel:
    def __init__(self, model):
        self.model = model

    def predict(self, dataset):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.predict(dataset)

    def evaluate(self, dataset, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            y_pred = self.predict(dataset)

            if Settings.estimation_type is EstimationType.regression:
                error = metrics.mean_squared_error(labels, y_pred)
            elif Settings.estimation_type is EstimationType.classification:
                y_pred = [round(value) for value in y_pred]
                error = metrics.accuracy_score(labels, y_pred)
            else:
                TypeError("Estimation task not recognized")

            return error

    def fit(self, boosting_matrix, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.fit(boosting_matrix, labels)

