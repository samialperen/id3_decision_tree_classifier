# This script contains user defined exceptions

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ProbabilityError(Error):
    """Exception raised for errors in the output of calc_prob func"""


