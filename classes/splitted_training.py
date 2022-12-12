from classes.pattern_boosting import PatternBoosting
from settings import Settings
import math


class SplittedTraining:
    def __init__(self):
        self.train_dataset = None
        self.test_dataset = None

    def main_train_splitted(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        settings = Settings()

        number_of_iterations = math.ceil(settings.maximum_number_of_steps / settings.training_batch_size)

        settings.maximum_number_of_steps = settings.training_batch_size

        pattern_boosting = PatternBoosting(settings)
        for i in range(number_of_iterations):
            pattern_boosting.training(self.train_dataset, self.test_dataset)


