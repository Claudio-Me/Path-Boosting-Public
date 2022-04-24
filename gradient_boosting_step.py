from settings import Settings
from R_code.interface_with_R_code import LaunchRCode
import random


class GradientBoostingStep:
    def select_column(self, matrix):
        """It makest one step of gradient boosting ant it returns the selected column"""
        if Settings.use_R is True:
            return self.__step_using_r(matrix)

        else:
            return self.__step_using_python(matrix)

    def __step_using_r(self, matrix):
        r_interface = LaunchRCode()
        selected_column_number = r_interface.launch_function(matrix)
        return selected_column_number

    def __step_using_python(self, matrix):
        print("warning __step_using_python is note implemented (it returns a random column)")
        return random.randint(0, len(matrix[0]) - 1)
