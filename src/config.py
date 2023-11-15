import copy
import os
import pathlib

import yaml

# Read the content of the configuration file.
DIR_PATH = pathlib.Path(__file__).parent
FILENAME = "config.yaml"
FILEPATH = os.path.join(DIR_PATH, FILENAME)
YAML_DOC = open(FILEPATH)
CONFIG = yaml.safe_load(YAML_DOC)
YAML_DOC.close()


def get(option: str):
    """
    Get the value of the configuration option.

    :param option: Single option or option's path separated by dots.

    Examples:
    >>> get('myoption')
    >>> get('mysection.mysubsection.myoption')

    :returns: The value of the option.

    """
    config = copy.deepcopy(CONFIG)
    keys = option.split(".")
    for key in keys:
        config = config[key]
    return config
