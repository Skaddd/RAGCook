import yaml
import os
from glob import glob
from typing import Dict, Union


def global_loading_configuration(
    configuration_dir: str,
) -> Dict[str, Union[int, float, str]]:
    """Fusion all configuration files.

    This function aims at unifying all configuration
    files present within a folder. This segementation of
    configuration files helps for the global readability.
    Args:
        configuration_dir (str): directory containing
        all configuration files.

    Returns:
        Dict[str, Union[int, float, str]]: Global configuration file.
    """

    global_configuration = {}
    for config_file in glob(os.path.join(configuration_dir, "*.yml")):

        with open(config_file, "r") as config_yml:
            global_configuration.update(yaml.safe_load(config_yml))

    return global_configuration
