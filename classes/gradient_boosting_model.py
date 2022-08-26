from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn import metrics
from settings import Settings
from classes.enumeration.estimation_type import EstimationType
from classes.enumeration.model_type import ModelType
import numpy as np
from R_code.interface_with_R_code import LaunchRCode


class GradientBoostingModel:
    def __init__(self, model):

        # note: this two 'if' are useless since they are doing the same operation, I leave them there just in case I want to modify the code later
        if isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
            self.model = model
        elif model is ModelType.r_model:
            self.model = model

    def predict(self, dataset):
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.predict(dataset)
        elif self.model is ModelType.r_model:
            r_predict_model = LaunchRCode(Settings.r_code_location, "main_predict")
            TypeError("prediction for R not implemented yet")
            prediction = r_predict_model.r_function(np.array(dataset), Settings.r_model_name)
            return prediction

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

        elif self.model is ModelType.r_model:
            y_pred = self.predict(dataset)
            model_error = metrics.mean_squared_error(labels, y_pred)
        return model_error

    def fit(self, boosting_matrix, labels):
        # ----------------------------------------------------------------------------------------------------------

        # N.B. this function returns the model in the case of XGB classifier, the selected column in case of R code

        # ----------------------------------------------------------------------------------------------------------
        if isinstance(self.model, XGBClassifier) or isinstance(self.model, XGBRegressor):
            return self.model.fit(boosting_matrix, labels)
        elif self.model is ModelType.r_model:
            r_select_column_and_train_model = LaunchRCode(Settings.r_code_location, "select_column")
            selected_column_number = r_select_column_and_train_model.r_function(np.array(boosting_matrix.matrix),
                                                                                np.array(labels),
                                                                                Settings.r_model_name,
                                                                                Settings.family)
            return selected_column_number
