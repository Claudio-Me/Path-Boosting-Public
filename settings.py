from classes.enumeration.estimation_type import EstimationType


class Settings:
    maximum_number_of_steps = 100  # call it maximum number of steps

    n_estimators = 1000

    use_R = True
    graph_label_variable = "target_svp_homo_lumo_gap"
    # graph_label_variable = "target_tzvp_homo_lumo_gap"

    estimation_type = EstimationType.regression
    # estimation_type = EstimationType.classification

    test_size = 0.2

    # the direcroty is relative to the python file location
    r_code_location = 'R_code/m_boost.R'

    # name of the file .RData where the model is saved
    r_model_name = "my_r_model"

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

    # quantity not used yet
    testing = False
    evaluate_test_dataset_during_training = True
    r_mboost_model_location = 'R_code/m_boost_model'

    r_function_name = 'main'
