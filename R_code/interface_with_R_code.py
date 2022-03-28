'''import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri


class LaunchRCode:
    def __init__(self):

        # activate automatic conversion to numpy arrays
        numpy2ri.activate()

        # Defining the R script and loading the instance in Python
        r = robjects.r
        r['R_code']('gradient_boost_step.R')

        # Loading the function we have defined in R.
        self.function_name = robjects.globalenv['function_name']

    def launch_function(self, input):

        # converting it into r object for passing into r function

        # Invoking the R function and getting the result
        result = self.function_name(input)

        return result
'''




class LaunchRCode:
    def __init__(self):

        pass

    def launch_function(self, input):


        return 0