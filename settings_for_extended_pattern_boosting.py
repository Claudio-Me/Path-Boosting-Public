class SettingsExtendedPatternBoosting:
    def __init__(self):
        self.plot_analysis = True
        self.n_estimators = 40
        self.show_tree: bool = False

        self.main_xgb_parameters = {'n_estimators': 1,
                                    'max_depth': 2,
                                    'learning_rate': 0.3,
                                    "eval_metric": "rmse",
                                    "objective": 'reg:squarederror',
                                    "reg_lambda": 0,
                                    "alpha": 0,
                                    "random_state": 0,
                                    'booster': 'gbtree'  # 'gbtree'  'gblinear'
                                    }

        self.choose_column_xgb_parameters = {'n_estimators': 1,
                                             'booster': 'gbtree',  # gbtree # gblinear
                                             'learning_rate': 0.3,
                                             "eval_metric": "rmse",
                                             "objective": 'reg:squarederror',
                                             "reg_lambda": 0,
                                             "alpha": 0,
                                             'max_depth': 1,
                                             'random_state': 0,
                                             'gamma': 0,
                                             'num_class': 1,


                                             'num_parallel_tree': 1
                                             }

        xgb_model_parameters = {
            'n_estimators': 1,
            'booster': 'gbtree',  # gbtree # gblinear
            'learning_rate': 0.3,
            "eval_metric": "rmse",
            "objective": 'reg:squarederror',
            "reg_lambda": 0,
            "alpha": 0

        }

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
