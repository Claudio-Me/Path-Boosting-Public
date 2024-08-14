from classes.enumeration.estimation_type import EstimationType
import platform
import pandas as pd
import os
import multiprocessing as mp


class Settings:
    maximum_number_of_steps = 300

    save_analysis = True
    show_analysis = True

    dataset_name = "5k_synthetic_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    generate_new_dataset = True

    # in the error graph Print only the last N learners
    tail = maximum_number_of_steps-150

    wrapper_boosting = False

    # used in wrapped boosting to specify the centers over which split the dataset
    considered_metal_centers = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # first block
                                    39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # second block
                                    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,  # lanthanides
                                    72, 73, 74, 75, 76, 77, 78, 79, 80,  # third block
                                    89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,  # actinides
                                    104, 105, 106, 107, 108, 109, 110, 111, 112]


    # do not expand if the paths are longer than this amount
    max_path_length = 101

    # portion of the whole dataset that needs to be used as test dataset
    test_size = 0.2

    scenario = 1

    target_train_error = 0.0000001

    synthetic_dataset_scenario = 1

    # it works only if "algorithm" is Xgb_step
    update_features_importance_by_comparison = True

    max_number_of_cores = mp.cpu_count()

    xgb_model_parameters = {
        'n_estimators': 1,
        'booster': 'gbtree',  # gbtree # gblinear
        'learning_rate': 0.3,
        "eval_metric": "rmse",
        "objective": 'reg:squarederror',
        "reg_lambda": 0,
        "alpha": 0

    }

    # note in gradient_boosting_model "eval_metric" is assumed to be rmse, be careful when changing it

    if xgb_model_parameters['booster'] == 'gblinear':
        xgb_model_parameters['updater'] = 'coord_descent'  # shotgun
        xgb_model_parameters['feature_selector'] = 'greedy'  # cyclic # greedy # thrifty
        # xgb_model_parameters['top_k'] = 1

    else:
        xgb_model_parameters['max_depth'] = 1
        xgb_model_parameters['gamma'] = 0

    plot_tree = False

    n_of_paths_importance_plotted: int = 30

    noise_variance = 0.2

    random_split_test_dataset_seed = 1
    random_coefficients_synthetic_dataset_seed = 1
    parallelization = False

    algorithm = "Xgb_step"  # "Full_xgb" "R" "Xgb_step"

    graph_label_variable = "target_tzvp_homo_lumo_gap"

    estimation_type = EstimationType.regression
    # estimation_type = EstimationType.classification

    # measure used for checkin the final error of the model (to plot error graphs)
    final_evaluation_error = "MSE"  # "absolute_mean_error" "MSE"

    # the direcroty is relative to the python file location
    r_code_relative_location = 'R_code/m_boost.R'

    # Base Learner used by mboost
    r_base_learner_name = "bols"  # "Gaussian", “bbs”, “bols”, “btree”, “bss”, “bns”

    verbose = True

    # Possible family names for loss function in R mode
    family = "Gaussian"
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
    r_model_name = "my_r_model"
    if True:
        r_model_name = r_base_learner_name + family + str(maximum_number_of_steps) + str(tail)

    # quantity not used yet

    multiple_training = True
    training_batch_size = 10

    testing = False
    evaluate_test_dataset_during_training = True
    n_estimators = 20

    # r_mboost_model_location = 'R_code/m_boost_model'

    @staticmethod
    def neg_gradient(y, y_hat):
        return (y - y_hat)

    pd.set_option('display.max_columns', None)



    scenario_1 = list({(28,), (28, 7), (28, 7, 6)})
    scenario_2 = list({(28, 7, 6, 6, 6, 35), (28, 7, 6, 6, 6), (28, 7, 6, 6), (28, 7, 6)})

    scenario_3 = [(57, 7, 7), (57, 7), (57,),
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

    if scenario == 1:
        target_paths = scenario_1
    elif scenario == 2:
        target_paths = scenario_2
    elif scenario == 3:
        target_paths = scenario_3

    @staticmethod
    def set_scenario(scenario):
        Settings.scenario= scenario
        if scenario == 1:
            Settings.target_paths = Settings.scenario_1
        elif scenario == 2:
            Settings.target_paths = Settings.scenario_2
        elif scenario == 3:
            Settings.target_paths = Settings.scenario_3