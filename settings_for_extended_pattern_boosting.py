
class SettingsExtendedPatternBoosting:
    def __init__(self):
        self.plot_analysis=True
        self. n_estimators=100

        self.xgb_parameters = {'n_estimators': 10,
                      'depth': 3,
                      'learning_rate': 0.3,
                      "eval_metric": "rmse",
                      "objective": 'reg:squarederror',
                      "reg_lambda": 0,
                      "alpha": 0,
                      "random_state": 0,
                      'booster': 'gbtree' # 'gbtree'  'gblinear'
                      }

        #if self.xgb_parameters['booster'] == 'gblinear':
            #self.xgb_parameters['updater'] = 'coord_descent'  # shotgun
            #self.xgb_parameters['feature_selector'] = 'greedy'  # cyclic # greedy # thrifty
            # xgb_model_parameters['top_k'] = 1

        #else:
            #self.xgb_parameters['max_depth'] = 1
            #self.xgb_parameters['gamma'] = 0

    def __repr__(self):
        attrs = vars(self)
        return ', '.join(f"{key}={value!r}" for key, value in attrs.items())

