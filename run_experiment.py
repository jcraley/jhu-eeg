import sys

import utils.pathmanager as pm
import utils.testconfiguration as tc
from utils.pipeline import Pipeline


def main():
    """Set up the experiment, initialize folders, and write config"""
    # Load the experiment configuration and paths
    argv = sys.argv[1:]
    params = tc.TestConfiguration('default.ini', argv)
    paths = pm.PathManager(params)
    paths.initialize_experiment_folders()
    pipeline = Pipeline(params, paths)

    # Save experiment configuration
    pipeline.write_config_file()

    # Load the datsets
    pipeline.initialize_val_dataset()
    pipeline.initialize_train_dataset()

    """Train or load a model"""
    if pipeline.params['load model fn']:
        pipeline.load_model()
    else:
        pipeline.initialize_model()
        pipeline.train()

    """Score test and train sets"""
    if params['score train']:
        pipeline.score_train_dataset()

    if params['score val']:
        pipeline.score_val_dataset()


if __name__ == '__main__':
    main()
