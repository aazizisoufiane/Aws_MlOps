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
import argparse
from sklearn.model_selection import train_test_split


def load_file():
    """
    Load a CSV file specified in the configuration.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """

    input_data_path = os.path.join("/opt/ml/processing/input", "train.csv")
    logger.info(f"Load file: {input_data_path}")

    df = pd.read_csv(input_data_path)
    return df.head(1000)


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


def main(split_ratio):
    """
    Main function to execute the data loading and preprocessing.

    Returns:
        pd.DataFrame: The final preprocessed DataFrame.
    """
    logger.info('Run the main function')

    df = load_file()
    df = preprocess(df)
    # Split the DataFrame into training and test sets
    train, test = train_test_split(df, test_size=split_ratio, random_state=42)
    labels_df = pd.DataFrame({"labels": list(labels)})

    train_labels_output_path = os.path.join("/opt/ml/processing/train", "train.csv")
    test_labels_output_path = os.path.join("/opt/ml/processing/test", "test.csv")
    labels_output_path = os.path.join("/opt/ml/processing/test", "labels.csv")
    logger.info("Saving training labels to {}".format(train_labels_output_path))
    train.to_csv(train_labels_output_path, index=False)
    logger.info("Saving test labels to {}".format(test_labels_output_path))
    test.to_csv(test_labels_output_path, index=False)
    labels_df.to_csv(labels_output_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split-ratio", type=float, default=0.3)
    args, _ = parser.parse_known_args()

    logger.info("Received arguments {}".format(args))
    main(split_ratio=args.train_test_split_ratio)
