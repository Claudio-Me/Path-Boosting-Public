import xgboost as xgb
import matplotlib.pyplot as plt


class XgbModelForExtendedPathBoosting:
    def __init__(self, **kwargs):
        # we expect each model that is passed to have the following methods:
        # .fit
        # .predict
        # .refit
        # .get_model
        # .get_dump
        # .plot_tree

        self.train_error = []
        self.test_error = []
        self.model: xgb.XGBRegressor = xgb.XGBRegressor(**kwargs)
        self.__fitted=False

    def fit(self, X, y, eval_set=None):
        # note: in case eval_set is passed, we assume the first parameter is the training dataset.
        if self.__fitted is False:
            self.model.fit(X, y, eval_set=eval_set)
            self.__fitted=True
        else:
            self.model.fit(X, y, eval_set=eval_set, xgb_model=self.model)

        self.train_error += self.model.evals_result()['validation_0']['rmse']
        try:
            self.test_error += self.model.evals_result()['validation_1']['rmse']
        except:
            pass
        return self

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)



    def get_dump(self):
        return self.model.get_booster().get_dump()

    def get_model(self):
        return self.model

    def plot_tree(self, num_trees: int=-1):
        xgb.plot_tree(self.model.get_booster(), num_trees=-1)
        plt.show()


