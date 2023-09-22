import yaml
from omegaconf import DictConfig

_config_name = 'orchestrator.yaml'


def get_config(config_name: str):
    """
    Load a YAML configuration file and return it as a DictConfig.

    Args:
        config_name (str): The name of the configuration file.

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.
    """
    # Construct the full path to the configuration file
    config_file = f"config/{config_name}"

    # Load and parse the YAML configuration file
    with open(config_file, 'r') as file:
        conf = yaml.safe_load(file)

    return DictConfig(conf)


# Load the configuration file
config = get_config(_config_name)
