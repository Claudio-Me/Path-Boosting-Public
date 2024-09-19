import xgboost


class InterfaceExtendedPathBoostingModel:
    def __init__(self, algorithm: str, model=None,**kwargs ):
        # we expect each model that is passed to have the following methods:
        # .fit
        # .predict

        self.algorithm = algorithm
        self.train_error=[]
        self.test_error=[]
        if model is not None:
            self.model = model
        else:

            self.algorithm = algorithm
            if algorithm == "xgboost":
                self.model = xgboost.XGBRegressor(**kwargs)
            else:
                raise ValueError(f"Algorithm {algorithm} is not supported")

    def fit(self, x,y, **kwargs):
        self.model.fit(X=x,y=y,**kwargs)
        self.train_error +=self.model.get_train_error_for_last_fitting_round()
        test_error = self.model.get_test_error_for_last_fitting_round()
        if test_error is not None:
            self.test_error += test_error
        return self.model



