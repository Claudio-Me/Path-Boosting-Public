from classes.enumeration.estimation_type import EstimationType


class Settings:
    maximum_number_of_steps = 100 # call it maximum number of steps

    n_estimators = 1000

    use_R = False
    graph_label_variable = "dasdadad" #target_svp_homo_lumo_gap"

    #estimation_type = EstimationType.regression
    estimation_type = EstimationType.classification

    test_size = 0.2

    # quantity not used yet
    testing = False
    evaluate_test_dataset_during_training = True
