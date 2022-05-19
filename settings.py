from classes.enumeration.estimation_type import EstimationType


class Settings:
    number_of_learners = 20

    use_R = False
    graph_label_variable = "target_svp_homo_lumo_gap"

    estimation_type = EstimationType.regression

    # quantity not used yet
    testing = False
