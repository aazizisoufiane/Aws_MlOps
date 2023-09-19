from omegaconf import DictConfig
import yaml
import os

_config_name = 'preprocess.yaml'

def get_config(config_name: str):
    """
    Load a YAML configuration file and return it as a DictConfig.

    Args:
        config_name (str): The name of the configuration file.

    Returns:
        DictConfig: The loaded configuration as a DictConfig object.
    """
    # Construct the full path to the configuration file
    config_file = f"code/config/{config_name}"
    # config_file = f"/opt/ml/processing/input/config/{config_name}"

    # Load and parse the YAML configuration file
    with open(config_file, 'r') as file:
        conf = yaml.safe_load(file)
    
    return DictConfig(conf)

# Load the configuration file
config = get_config(_config_name)

# Define a list of labels
selected_columns = ['ABSTRACT']
labels = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

