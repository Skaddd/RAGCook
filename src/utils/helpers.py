import hashlib
import logging
import os
from glob import glob
from pathlib import Path
from typing import Dict, Union

import yaml
import yaml.parser

# could be a env variable later
CONFIG_DIR = Path(__file__).parent.parent.parent / "conf"

logger = logging.getLogger(__name__)


def generate_hash(text_content: str) -> str:
    """Generate unique identifier from piece of text."""

    return hashlib.sha256(text_content.encode()).hexdigest


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
        try:
            with open(config_file, "r") as config_yml:
                global_configuration.update(yaml.safe_load(config_yml))

        except yaml.parser.ParserError:
            logger.warning(
                f" Parsing error in the following file : {config_file}"
            )

    return global_configuration


def load_config():
    """Loads global configuration."""
    print(CONFIG_DIR)
    return global_loading_configuration(CONFIG_DIR)
