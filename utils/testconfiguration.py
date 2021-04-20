import argparse
import configparser
import json


def get_config_fn_from_argv(argv):
    """Check the arguments for a config file and return it, or None"""
    if '--config_fn' in argv:
        config_fn = argv[argv.index('--config_fn') + 1]
        argv.remove('--config_fn')
        argv.remove(config_fn)
        return config_fn
    else:
        return None


class TestConfiguration():

    def __init__(self, default_cfg='default.ini', argv=None):
        """Load the configuration from the given config_fn"""

        # Read the default settings
        self.config = configparser.ConfigParser()
        self.config.read(default_cfg)
        # Check for a command line config argument
        if argv is not None and '--config_fn' in argv:
            # Get the config filename
            config_fn = argv[argv.index('--config_fn') + 1]
            argv.remove('--config_fn')
            argv.remove(config_fn)
            # Read the new config file
            new_cfg = configparser.ConfigParser()
            new_cfg.read(config_fn)
            # Overwrite the default values
            self._overwrite_with_config(new_cfg)
        if argv is not None:
            self._parse_arguments(argv)
        self._update()

    def _overwrite_with_config(self, new_cfg):
        """Given a new, possibly less specific config file, overwrite
        defaults"""
        for section in new_cfg.sections():
            for key, val in new_cfg.items(section):
                self.config.set(section, key, val)

    def _overwrite_with_kwargs(self, **overwrite):
        # Override default config params if any kwargs were passed
        for key, value in overwrite.items():
            new_key = key.replace('_', ' ')
            new_val = str(value).replace('~', ' ')
            for section in self.config.sections():
                if new_key in self.config[section]:
                    self.config.set(section, new_key, new_val)
                    break

    def _update(self):
        """Update dictionary of internal parameters"""
        self.all_params = {}
        self._update_experiment_params()
        self._update_preprocessing_params()
        self._update_model_params()

    def _update_experiment_params(self):
        # EXPERIMENT parameters
        exp_cfg = self.config['EXPERIMENT']
        self.all_params.update({
            'experiment name': exp_cfg['experiment name'],
            'trial name': exp_cfg['trial name'],
            'dataset': exp_cfg['dataset'],
            'train manifest': exp_cfg['train manifest'],
            'val manifest': exp_cfg['val manifest'],
            'seed': exp_cfg.getint('seed'),
            'channel list': exp_cfg['channel list'],
            'score val': exp_cfg.getboolean('score val'),
            'visualize val': exp_cfg.getboolean('visualize val'),
            'score train': exp_cfg.getboolean('score train'),
            'visualize train': exp_cfg.getboolean('visualize train'),
            'smoothing': exp_cfg.getint('smoothing'),
            'features': json.loads(exp_cfg['features']),
            'load to device': exp_cfg.getboolean('load to device'),
            'load as': exp_cfg['load as'],
            'load model fn': exp_cfg['load model fn'],
            'fps per hour': exp_cfg.getfloat('fps per hour'),
            'fp time per hour': exp_cfg.getfloat('fp time per hour'),
            'max samples before sz': exp_cfg.getint('max samples before sz'),
            'count post sz': exp_cfg.getboolean('count post sz'),
        })

    def _update_preprocessing_params(self):
        # PREPROCESSING parameters
        preprocessing_cfg = self.config['PREPROCESSING']
        self.all_params.update({
            'notch': preprocessing_cfg.getboolean('notch'),
            'lpf fc': preprocessing_cfg.getfloat('lpf fc'),
            'hpf fc': preprocessing_cfg.getfloat('hpf fc'),
            'clip level': preprocessing_cfg.getfloat('clip level'),
            'normalize': preprocessing_cfg.getboolean('normalize'),
            'window length': preprocessing_cfg.getfloat('window length'),
            'overlap': preprocessing_cfg.getfloat('overlap'),
        })

    def _update_model_params(self):
        # MODEL parameters
        model_cfg = self.config['MODEL']
        self.all_params.update({
            'model type': model_cfg['model type'],
            'model kwargs': model_cfg['model kwargs'],
        })

    def _parse_arguments(self, argv):
        """Parse the command line arguments"""
        parser = argparse.ArgumentParser()
        for section in self.config.sections():
            for key in self.config[section]:
                arg_name = '--' + key.replace(' ', '_').lower()
                parser.add_argument(arg_name)
        override_kwargs = vars(parser.parse_args(argv))
        override_kwargs = {k: v for k,
                           v in override_kwargs.items() if v is not None}
        self._overwrite_with_kwargs(**override_kwargs)

    def write(self, fn):
        """Write settings to a new config file"""
        with open(fn, 'w') as f:
            self.config.write(f)

    def __getitem__(self, key):
        return self.all_params[key]

    def __setitem__(self, key, value):
        self.all_params[key] = value

    def __contains__(self, key):
        return key in self.all_params
