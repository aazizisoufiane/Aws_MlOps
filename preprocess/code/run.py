import sys
import os
import subprocess
subprocess.call([sys.executable, "-m", "pip", "install", "-r", "/opt/ml/processing/input/code/requirements.txt"])
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))))

from config import config, selected_columns, labels
from omegaconf import DictConfig
import pandas as pd
import yaml
from logger import logger

def load_file():
    """
    Load a CSV file specified in the configuration.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    logger.info(f"Load file: {config.path.train}")
    df = pd.read_csv(config.path.train)
    return df

def preprocess(df):
    """
    Preprocess the input DataFrame by selecting specific columns and transforming labels.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    logger.info('Preprocess data')
    # Select specific columns
    
    df = df[selected_columns + labels]
    
    # Transform labels into lists
    df['labels'] = df[labels].apply(lambda row: row.tolist(), axis=1)

    return df

def main():
    """
    Main function to execute the data loading and preprocessing.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame.
    """
    logger.info('Run the main function')

    df = load_file()
    df = preprocess(df)
    return df
