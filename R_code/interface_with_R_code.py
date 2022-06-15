import pandas as pd

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri


class LaunchRCode:
    def __init__(self):
        # activate automatic conversion to numpy arrays
        numpy2ri.activate()

        pi = robjects.r('pi')
        print(pi)
        # Defining the R script and loading the instance in Python
        r = robjects.r
        r['source']('R_code/r_code_test.R')

        # Loading the function we have defined in R.
        self.function_name = robjects.globalenv['new_function']

    def launch_function(self, matrix, labels):
        # converting it into r object for passing into r function

        # Invoking the R function and getting the result
        result = self.function_name(matrix, labels)

        return result
