import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd

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


    def fit(self, X, y, eval_set):
        # we assume y is the real target and we compute the negative gradient
        # note: in case eval_set is passed, we assume the first parameter is the training dataset.
        # create new base learner model
        new_base_learner = xgb.XGBRegressor(**self.__kwargs)


        if len(self.base_learners_list) == 0:
            # it is the first time we fit it so we do not need to compute the neg gradient
            new_base_learner.fit(X, y, eval_set=eval_set)
            original_dataset = eval_set[0][0]
            self.__last_prediction = pd.Series(new_base_learner.predict(original_dataset))
            self.base_learners_list.append(new_base_learner)


        else:

            # compute neg gradient
            # we assume kwargs eval_set[0][1] are the original labels
            neg_gradient = self.__neg_gradient(y= eval_set[0][1], y_hat=self.__last_prediction)
            neg_gradient = pd.Series(neg_gradient).loc[X.index]
            new_base_learner = xgb.XGBRegressor(**self.__kwargs)
            new_base_learner.fit(X, neg_gradient, eval_set=eval_set)
            # we assume this is the original training dataset
            original_dataset=eval_set[0][0]
            self.__last_prediction += new_base_learner.predict(original_dataset)
            self.base_learners_list.append(new_base_learner)

        # add the errors
        self.train_error += new_base_learner.evals_result()['validation_0']['rmse']
        try:
            self.test_error += new_base_learner.evals_result()['validation_1']['rmse']
        except:
            pass
        return self



    def predict(self, X, **kwargs):
        prediction =[]
        for base_learner in self.base_learners_list:
            prediction.append(base_learner.predict(X, **kwargs))

        return sum(prediction)



    def get_dump(self):

        return [base_learner.get_booster().get_dump()for base_learner in self.base_learners_list]

    def get_model(self):
        return self.base_learners_list

    def plot_tree(self):
        for base_learner in self.base_learners_list:
            xgb.plot_tree(base_learner.get_booster())
            plt.show()

    @staticmethod
    def __neg_gradient(y, y_hat):
        return 2 * (y - y_hat)
