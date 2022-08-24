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

    #the direcroty is relative to the python file location
    r_code_location = 'R_code/m_boost.R'

    # name of the file .RData where the model is saved
    r_model_name = "my_r_model"


    # quantity not used yet
    testing = False
    evaluate_test_dataset_during_training = True
    r_mboost_model_location = 'R_code/m_boost_model'

    r_function_name = 'main'
