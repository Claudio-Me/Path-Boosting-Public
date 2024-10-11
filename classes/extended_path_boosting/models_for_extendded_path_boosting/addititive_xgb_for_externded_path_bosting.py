import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
from classes.extended_boosting_matrix import ExtendedBoostingMatrix


class AdditiveXgbForExtendedPathBosting:
    def __init__(self, **kwargs):
        # we expect each model that is passed to have the following methods:
        # .fit
        # .predict
        # .refit
        # .get_model
        # .get_dump
        # .plot_tree

        self.__kwargs = kwargs
        self.__last_prediction: pd.Series = None
        self.train_error = []
        self.test_error = []
        self.base_learners_list: list[xgb.XGBRegressor] = []

    def fit(self, X, y,best_path,dict_of_interaction_constraints, eval_set = None, negative_gradient=None):
        # we assume y is the real target and we compute the negative gradient
        # note: in case eval_set is passed, we assume the first parameter is the training dataset.
        # create new base learner model

        zeroed_x_df, zeroed_y = ExtendedBoostingMatrix.zero_all_elements_except_the_ones_referring_to_path(
            x_df=X, y=y, path=best_path, dict_of_interaction_constraints=dict_of_interaction_constraints)

        new_base_learner = xgb.XGBRegressor(**self.__kwargs)

        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient
            new_base_learner.fit(zeroed_x_df, zeroed_y, eval_set=eval_set)
            self.__last_prediction = pd.Series(new_base_learner.predict(X))
            self.base_learners_list.append(new_base_learner)


        else:

            # compute the new target (we have to use zeroed_y - true_neg_gradient instead of just zeroed_y, more explained in paper)
            if negative_gradient is None:
                negative_gradient = self.__neg_gradient(y=y, y_hat=self.__last_prediction)
            new_y = pd.Series(negative_gradient).loc[zeroed_y.index]

            new_base_learner = xgb.XGBRegressor(**self.__kwargs)
            new_base_learner.fit(zeroed_x_df, new_y, eval_set=eval_set)
            self.__last_prediction += new_base_learner.predict(X)
            self.base_learners_list.append(new_base_learner)

        # add the errors
        self.train_error.append(pow(new_base_learner.evals_result()['validation_0']['rmse'][-1],2))
        try:
            self.test_error.append(pow(new_base_learner.evals_result()['validation_1']['rmse'][-1],2))
        except:
            pass
        return self

    def predict(self, X, **kwargs):
        prediction = []
        for base_learner in self.base_learners_list:
            prediction.append(base_learner.predict(X, **kwargs))

        return sum(prediction)

    def get_dump(self):

        return [base_learner.get_booster().get_dump() for base_learner in self.base_learners_list]

    def get_model(self):
        return self.base_learners_list

    def plot_tree(self):
        for base_learner in self.base_learners_list:
            xgb.plot_tree(base_learner.get_booster())
            plt.show()

    @staticmethod
    def __neg_gradient(y, y_hat):
        return 2 * (y - y_hat)
