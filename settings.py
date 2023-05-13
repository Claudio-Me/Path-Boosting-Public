from classes.enumeration.estimation_type import EstimationType
import platform
import pandas as pd
import os


class Settings:
    maximum_number_of_steps = 12  # call it maximum number of steps

    save_analysis = False
    show_analysis = True

    dataset_name = "60k_dataset"  # "5k_synthetic_dataset" "5_k_selection_graphs"  "60k_dataset"
    generate_new_dataset = False

    # in the error graph Print only the last N learners
    tail = 3800

    considered_metal_center=[46]


    # do not expand if the paths are longer than this amount
    max_path_length = 100

    target_train_error = 0.0001

    xgb_model_parameters = {
        'n_estimators': 1,
        'booster': 'gbtree',  # gbtree # gblinear
        'learning_rate': 0.3,
        "eval_metric": "rmse",
        "objective": 'reg:squarederror',
        "reg_lambda": 0
    }
    if xgb_model_parameters['booster'] == 'gblinear':
        xgb_model_parameters['updater'] = 'coord_descent'  # shotgun
        xgb_model_parameters['feature_selector'] = 'thrifty'  # cyclic # greedy # thrifty
        xgb_model_parameters['top_k'] = 1

    else:
        xgb_model_parameters['max_depth'] = 1

    plot_tree = False

    random_split_test_dataset_seed = 1
    random_coefficients_synthetic_dataset_seed=1

    parallelization = False

    algorithm = "Xgb_step"  # "Full_xgb" "R" "Xgb_step"

    graph_label_variable = "target_tzvp_homo_lumo_gap"

    estimation_type = EstimationType.regression
    # estimation_type = EstimationType.classification

    # measure used for checkin the final error of the model (to plot error graphs)
    final_evaluation_error = "MSE"  # "absolute_mean_error" "MSE"

    # portion of the whole dataset that needs to be used as test dataset
    test_size = 0.2

    # the direcroty is relative to the python file location
    r_code_relative_location = 'R_code/m_boost.R'

    # Base Learner used by mboost
    r_base_learner_name = "bols"  # "Gaussian", “bbs”, “bols”, “btree”, “bss”, “bns”

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
