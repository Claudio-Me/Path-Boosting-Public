import numpy as np

from settings import Settings
import pandas as pd
if Settings.algorithm=="R":
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import numpy2ri


class LaunchRCode:
    def __init__(self, function_location, function_name):
        # activate automatic conversion to numpy arrays
        numpy2ri.activate()
        self.load_function(function_location, function_name)

    def load_function(self, function_location, function_name):
        # Defining the R script and loading the instance in Python
        r = robjects.r
        r['source'](function_location)

        # Loading the function we have defined in R.
        self.r_function = robjects.globalenv[function_name]

    def launch_function(self, arguments):
        # converting it into r object for passing into r function

        # Invoking the R function and getting the result
        result = self.r_function(np.array(arguments[0]), np.array(arguments[1]), arguments[2])

        return result

    def testing(self):
        pi = robjects.r('pi')
        print(pi)
