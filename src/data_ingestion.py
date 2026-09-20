"""
data_ingestion.py

This module handles the data ingestion stage of the ML pipeline:
    1. Loading parameters from params.yaml
    2. Loading raw data from a source (CSV/URL)
    3. Preprocessing the data (cleaning, renaming columns)
    4. Splitting the data into train and test sets
    5. Saving the processed data to disk

Logging is configured to track the pipeline's execution and capture 
any errors encountered along the way, both on the console and in a 
persistent log file.
"""

import os
import logging

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(LOG_DIR, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ------------------------------------------------------------------
# Parameter Loading
# ------------------------------------------------------------------

def load_params(params_path: str) -> dict:
    """Load the parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# ------------------------------------------------------------------
# Data Ingestion Functions
# ------------------------------------------------------------------

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file or URL into a pandas DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the raw dataset.
    """
    try:
        df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """
    Save the train and test datasets to a 'raw' subdirectory as CSV files.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# ------------------------------------------------------------------
# Main Pipeline Execution
# ------------------------------------------------------------------

def main():
    """
    Orchestrates the full data ingestion pipeline:
    load params -> load data -> preprocess -> split -> save.
    """
    try:
        # Load test_size from params.yaml instead of hardcoding it
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']

        data_source_url = "https://raw.githubusercontent.com/NiteshBishwas/Datasets/refs/heads/main/spam.csv"

        df = load_data(data_url=data_source_url)
        final_df = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=2
        )

        save_data(train_data, test_data, data_path='./data')

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()


    


