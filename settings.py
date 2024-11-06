from classes.enumeration.estimation_type import EstimationType
import platform
import pandas as pd
import os
import multiprocessing as mp
import random
import sys
from numbers import Number
from collections import deque
from collections.abc import Set, Mapping

# TODO delete this four imports
import sys
from types import ModuleType, FunctionType
from gc import get_referents
import pickle


class Settings:
    # -----------------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.maximum_number_of_steps = 500

        self.save_analysis: bool = True
        self.show_analysis: bool = False

        self.dataset_name = "60k_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
        self.generate_new_dataset = False
        self.generate_from_binary_file = True

        # convert the nx graphs to undirected graphs, used only when generating new dataset
        self.convert_to_undirected = True

        # in the error graph Print only the last N learners
        self.tail = self.maximum_number_of_steps + 1

        self.wrapper_boosting: bool = True

        self.noise_variance = 0.2

        # do not expand if the paths are longer than this amount
        self.max_path_length = 6

        # portion of the whole dataset that needs to be used as test dataset
        self.test_size = 0.2

        self.scenario = 3

        self.unique_id_name = "29001"

        self.target_train_error = 0.0000001

        self.plot_log_info: bool = True

        # -----------------------------------------------------------------------------------------------------------------

        # it works only if "algorithm" is Xgb_step
        self.update_features_importance_by_comparison = True
        self.verbose = True

        self.max_number_of_cores = mp.cpu_count()

        self.xgb_model_parameters = {
        'n_estimators': 1,
        'booster': 'gbtree',  # gbtree # gblinear
        'learning_rate': 0.3,
        "eval_metric": "rmse",
        "objective": 'reg:squarederror',
        "reg_lambda": 0,
        "alpha": 0

        }

    # note in gradient_boosting_model "eval_metric" is assumed to be rmse, be careful when changing it

        if self.xgb_model_parameters['booster'] == 'gblinear':
            self.xgb_model_parameters['updater'] = 'coord_descent'  # shotgun
            self.xgb_model_parameters['feature_selector'] = 'greedy'  # cyclic # greedy # thrifty
            # self.xgb_model_parameters['top_k'] = 1

        else:
            self.xgb_model_parameters['max_depth'] = 1
            self.xgb_model_parameters['gamma'] = 0

        self.base_learner_tree_parameters = {'max_depth': 1,
                  'splitter': "best",
                  'random_state': 2,
                  'criterion': "squared_error"
                  }



        self.plot_tree = False

        self.n_of_paths_importance_plotted: int = 30

        self.random_split_test_dataset_seed = 1
        self.random_coefficients_synthetic_dataset_seed = 1

        self.random_generator_for_noise_in_synthetic_dataset = random.Random()
        self.random_generator_for_noise_in_synthetic_dataset.seed(self.random_coefficients_synthetic_dataset_seed + 1)

        self.parallelization = False

        self.algorithm = "Xgb_step"  # "Full_xgb" "R" "Xgb_step" "decision_tree"

        self.graph_label_variable = "target_tzvp_homo_lumo_gap"

        self.estimation_type = EstimationType.regression
        # self.estimation_type = EstimationType.classification

        # measure used for checkin the final error of the model (to plot error graphs)
        self.final_evaluation_error = "MSE"  # "absolute_mean_error" "MSE"

        # the direcroty is relative to the python file location
        self.r_code_relative_location = 'R_code/m_boost.R'

        # Base Learner used by mboost
        self.r_base_learner_name = "bols"  # "Gaussian", “bbs”, “bols”, “btree”, “bss”, “bns”

        # Possible family names for loss function in R mode
        self.family = "Gaussian"
        # Gaussian: Gaussian
        # AdaExp: AdaExp
        # AUC: AUC()
        # Binomial: Binomial(type=c("adaboost", "glm"), link=c("logit", "probit", "cloglog", "cauchit", "log"), ...)
        # GaussClass: GaussClass()
        # GaussReg: GaussReg()
        # Huber: Huber(d=NULL)
        # Laplace: Laplace()
        # Poisson: Poisson()
        # GammaReg: GammaReg(nuirange=c(0, 100))
        # CoxPH: CoxPH()
        # QuantReg: QuantReg(tau=0.5, qoffset=0.5)
        # ExpectReg: ExpectReg(tau=0.5)
        # NBinomial: NBinomial(nuirange=c(0, 100))
        # PropOdds: PropOdds(nuirange=c(-0.5, -1), offrange=c(-5, 5))
        # Weibull: Weibull(nuirange=c(0, 100))
        # Loglog: Loglog(nuirange=c(0, 100))
        # Lognormal: Lognormal(nuirange=c(0, 100))
        # Gehan: Gehan()
        # Hurdle: Hurdle(nuirange=c(0, 100))
        # Multinomial: Multinomial()
        # Cindex: Cindex(sigma=0.1, ipcw=1)
        # RCG: RCG(nuirange=c(0, 1), offrange=c(-5, 5))

        # name of the file .RData where the model is saved
        self.r_model_name = "my_r_model"
        if True:
            self.r_model_name = self.r_base_learner_name + self.family + str(self.maximum_number_of_steps) + str(self.tail)

        # quantity not used yet

        self.multiple_training = True
        self.training_batch_size = 10

        self.testing = False
        self.evaluate_test_dataset_during_training = True
        self.n_estimators = 20

        # self.r_mboost_model_location = 'R_code/m_boost_model'



        pd.set_option('display.max_columns', None)

        # used in wrapped boosting to specify the centers over which split the dataset
        self.considered_metal_centers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                                39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                                57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                                72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                                89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                                104, 105, 106, 107, 108, 109, 110, 111, 112]

        self.scenario_1 = list({(28,), (28, 7), (28, 7, 6)})
        self.scenario_2 = list({(28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6), (28, 7, 6, 6), (28, 7, 6)})

        self.scenario_3 = [(57, 7, 7), (57, 7), (57,),
                  (72, 7, 14), (72, 7), (72,),
                  (78, 6, 7), (78, 6), (78,),
                  (47, 7, 7), (47, 7), (47,),
                  (74, 15, 8), (74, 15), (74,),
                  (80, 7, 7), (80, 7), (80,),
                  (77, 7, 7), (77, 7), (77,),
                  (40, 7, 14), (40, 7), (40,),
                  (21, 7, 14), (21, 7), (21,),
                  (27, 6, 5), (27, 6), (27,),
                  (27, 6, 8),
                  (42, 7, 7), (42, 7), (42,),
                  (39, 7, 7), (39, 7), (39,),
                  (39, 7, 14),
                  (39, 6, 7), (39, 6),
                  (39, 6, 14),
                  (39, 6, 5),
                  (45, 7, 7), (45, 7), (45,),
                  (48, 8, 7), (48, 8), (48,)]

        if self.scenario == 1:
            self.target_paths = self.scenario_1
        elif self.scenario == 2:
            self.target_paths = self.scenario_2
        elif self.scenario == 3:
            self.target_paths = self.scenario_3


    def set_scenario(self, scenario):
        self.scenario = scenario
        if scenario == 1:
            self.target_paths = self.scenario_1
        elif scenario == 2:
            self.target_paths = self.scenario_2
        elif scenario == 3:
            self.target_paths = self.scenario_3

    cross_validation_k_fold_seed = 5


    def print_principal_values(self):
        print("Settings Principal Values:")
        for attr, value in vars(self).items():
            if not attr.startswith('__') and not callable(value):
                if not isinstance(value, list):
                    print(f"{attr}: {value}")


    @staticmethod
    def log_principal_settings_values(logger, settings):
        logger.info("Settings Principal Values:")
        for attr, value in vars(settings).items():
            if not attr.startswith('__') and not callable(value):
                if not isinstance(value, list):
                    logger.info(f"{attr}: {value}")


    # TODO remove the method getsize
    @staticmethod
    def getsize(obj):
        """sum size of object & members."""
        BLACKLIST = type, ModuleType, FunctionType
        if isinstance(obj, BLACKLIST):
            raise TypeError('getsize() does not take argument of type: ' + str(type(obj)))
        seen_ids = set()
        size = 0
        objects = [obj]
        while objects:
            need_referents = []
            for obj in objects:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size / 1000000

    # TODO remove the method get_pickle_size

    @staticmethod
    def get_pickle_size(my_object):
        # Serialize the object into a bytes object
        serialized_object = pickle.dumps(my_object)

        # Get the size of the serialized bytes object
        size_in_bytes = len(serialized_object)
        # convert into mb
        return size_in_bytes / 1000000

    @staticmethod
    def neg_gradient(y, y_hat):
        return y - y_hat

