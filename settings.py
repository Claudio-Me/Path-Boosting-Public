from classes.enumeration.estimation_type import EstimationType


class Settings:
    number_of_learners = 20

    use_R = False
    graph_label_variable = "target_svp_homo_lumo_gap"

    estimation_type = EstimationType.regression
    test_size = 0.2

    # quantity not used yet
    testing = False
    evaluate_test_dataset_during_training = True
