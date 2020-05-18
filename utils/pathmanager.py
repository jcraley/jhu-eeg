import os


def init_folder(folder_path):
    """Initialize a folder"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


class PathManager():
    """A class for managing all of the paths for an experiment"""

    def __init__(self, config_dict):
        """
        Create all of the necessary folders and store the paths

        inputs:
            config_dict - a dictionary with the fields "experiment name",
                "trial name", "dataset"
        """
        self.config_dict = config_dict

        # Set up the experiment specific folders
        exp_path = os.path.join(
            'Experiments', self.config_dict['experiment name'])
        trial_path = os.path.join(exp_path, self.config_dict['trial name'])
        self.path_dict = {
            'exp': exp_path,
            'trial': trial_path,
            'figures': os.path.join(trial_path, 'figures'),
            'models': os.path.join(trial_path, 'models'),
            'results': os.path.join(trial_path, 'results'),
            'predictions': os.path.join(trial_path, 'predictions')
        }

        # Initialize data specific folders
        self.preprocessing_str = (
            "lpf_fc{}_hpf_fc{}_notch{}_clip{}_normalize{}".format(
                self.config_dict['lpf fc'], self.config_dict['hpf fc'],
                self.config_dict['notch'], self.config_dict['clip level'],
                self.config_dict['normalize']
            ))
        self.window_str = "window_length{}_overlap{}".format(
            self.config_dict['window length'], self.config_dict['overlap']
        )
        self.path_dict.update({
            'raw data': os.path.join('EDF', self.config_dict['dataset']),
            'buffers': os.path.join(
                'Buffers', self.config_dict['dataset'], self.preprocessing_str
            ),
            'data': os.path.join(
                'WindowedBuffers', self.config_dict['dataset'],
                self.preprocessing_str, self.window_str
            ),
            'labels': os.path.join(
                'Labels', self.config_dict['dataset'], self.window_str
            ),
            'features': os.path.join(
                'Features', self.config_dict['dataset'],
                self.preprocessing_str, self.window_str
            )
        })

    def initialize_experiment_folders(self):
        """Initialize folder tree for a given experiment"""
        # Initialize all of the folders
        init_folder(self.path_dict['exp'])
        init_folder(self.path_dict['trial'])
        init_folder(self.path_dict['figures'])
        init_folder(self.path_dict['models'])
        init_folder(self.path_dict['results'])
        init_folder(self.path_dict['predictions'])

    def initialize_folder(self, key):
        """Initialize a specific folder"""
        init_folder(self.path_dict[key])

    def add_feature_folder(self, key):
        """Add a feature folder"""
        self.path_dict['features'] = os.path.join(
            'Features', self.config_dict['dataset'], self.preprocessing_str,
            self.window_str)
        init_folder(self.path_dict['features'])
        self.path_dict[key] = os.path.join(self.path_dict['features'], key)
        init_folder(self.path_dict[key])

    def __getitem__(self, key):
        return self.path_dict[key]

    def __setitem__(self, key, value):
        self.path_dict[key] = value
        init_folder(self.path_dict[key])
