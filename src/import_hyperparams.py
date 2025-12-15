"""import_hyperparams.py"""
import json

def import_params(file):
    """
    Imports the hyperparams from the designated file for use in training program functions
    """
    with open(file, 'r') as param_file:
        hyperparameters = json.load(param_file)
        return hyperparameters
