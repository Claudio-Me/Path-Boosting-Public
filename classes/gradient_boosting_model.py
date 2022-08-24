from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from enumeration.model_type import ModelType
import numpy as np
from R_code.interface_with_R_code import LaunchRCode


class GradientBoostingModel:
    def __init__(self, model):

        # note: this two 'if' are useless since they are doing the same operation, I leave them there just in case I want to modify the code later
        if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
            self.model = model
        if model is ModelType.r_model:
            self.model = model

    def predict(self, dataset):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.predict(dataset)
        if self.model is ModelType.r_model:
            r_evaluate_model = LaunchRCode(Settings.r_code_location, "predict")
            TypeError("prediction for R not implemented yet")
            #prediction = r_evaluate_model.r_function(arguments...)

    def evaluate(self, dataset, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            y_pred = self.predict(dataset)

            if Settings.estimation_type is EstimationType.regression:
                model_error = metrics.mean_squared_error(labels, y_pred)
            elif Settings.estimation_type is EstimationType.classification:
                y_pred = [round(value) for value in y_pred]
                model_error = metrics.accuracy_score(labels, y_pred)
            else:
                TypeError("Estimation task not recognized")

            return model_error
        if self.model is ModelType.r_model:
            r_evaluate_model = LaunchRCode(Settings.r_code_location, "evaluate")
            TypeError("evaluation for R not implemented yet")
            #evaluation = r_evaluate_model.r_function(arguments...)

    def fit(self, boosting_matrix, labels):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.fit(boosting_matrix, labels)
