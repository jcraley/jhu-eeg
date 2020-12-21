import sys

import torch

import utils.testconfiguration as tc
import utils.pathmanager as pm
from utils.pipeline import Pipeline

torch.manual_seed(0)


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the experiment configuration and paths
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_experiment_folders()
    pipeline = Pipeline(params, paths)

    # Write config file
    pipeline.write_config_file()

    # Load the datsets
    # pipeline.initialize_train_dataset()
    # pipeline.initialize_val_dataset()

    # """Train or load a model"""
    # if pipeline.params['load model fn']:
    #     pipeline.load_model()

    """Score test and train sets"""
    pipeline.score_saved_train_predictions()
    pipeline.score_saved_val_predictions()


if __name__ == '__main__':
    main()
