

class SettingsExtendedPatternBoosting:
    def __init__(self):
        self.plot_analysis = True
        self.n_estimators = 20
        self.show_tree: bool = True
        self.xgb_verbose: bool = False
        self.name_model = 'additive_tree' # 'additive_xgboost' 'xgboost' 'additive_tree'

        self.main_xgb_parameters = {'n_estimators': 1,
                                    'max_depth': 2,
                                    'learning_rate': 0.3,
                                    "eval_metric": "rmse",
                                    "objective": 'reg:squarederror', # None 'reg:squarederror'
                                    "reg_lambda": 0,
                                    "alpha": 0,
                                    "random_state": 0,
                                    'booster': 'gbtree',  # 'gbtree'  'gblinear'
                                    }

        self.choose_column_xgb_parameters = {'n_estimators': 1,
                                             'booster': 'gbtree',  # gbtree # gblinear
                                             'max_depth': 1,
                                             'learning_rate': 0.3,
                                             #"eval_metric": "rmse",
                                             "objective": 'reg:squarederror',
                                             "reg_lambda": 0,
                                             "alpha": 0,
                                             'random_state': 0,
                                             }

        self.base_tree_parameters = {'max_depth': 2,
                                     'random_state': 0,
                                     'splitter': 'best',
                                     'criterion':"squared_error",}


        # if self.main_xgb_parameters['booster'] == 'gblinear':
        # self.main_xgb_parameters['updater'] = 'coord_descent'  # shotgun
        # self.main_xgb_parameters['feature_selector'] = 'greedy'  # cyclic # greedy # thrifty
        # self.main_xgb_model_parameters['top_k'] = 1

        # else:
        # self.main_xgb_parameters['max_depth'] = 1
        # self.main_xgb_parameters['gamma'] = 0

    def __repr__(self):
        attrs = vars(self)
        return ', '.join(f"{key}={value!r}" for key, value in attrs.items())
