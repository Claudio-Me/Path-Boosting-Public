from classes.enumeration.estimation_type import EstimationType
import platform

class Settings:
    maximum_number_of_steps = 50  # call it maximum number of steps

    # in the error graph Print only the last 20 learners
    tail = 50

    if platform.system() == 'WindowsD':
        graphs_folder = "C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/graphs"

    elif platform.system() == 'Darwin':
        pass

    algorithm = "Xgb_step"  # "Full_xgb", "R"

    # graph_label_variable = "target_svp_homo_lumo_gap"
    graph_label_variable = "target_tzvp_homo_lumo_gap"

    estimation_type = EstimationType.regression
    # estimation_type = EstimationType.classification

    test_size = 0.2

    # the direcroty is relative to the python file location
    r_code_relative_location = 'R_code/m_boost.R'

    r_model_location = "C:/Users/popcorn/Desktop/0/UiO/PhD/code/pattern_boosting/R_code"

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
