"""
data_ingestion.py

This module handles the data ingestion stage of the ML pipeline:
    1. Loading raw data from a source (CSV/URL)
    2. Preprocessing the data (cleaning, renaming columns)
    3. Splitting the data into train and test sets
    4. Saving the processed data to disk

Logging is configured to track the pipeline's execution and capture 
any errors encountered along the way, both on the console and in a 
persistent log file.
"""

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split



# Logging Configuration


# Ensure the "logs" directory exists to store log files
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Create a logger specific to the data ingestion module
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Handler to print logs to the console (useful during development)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Handler to persist logs to a file (useful for later debugging/auditing)
log_file_path = os.path.join(LOG_DIR, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define a consistent log message format: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# Data Ingestion Functions


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file or URL into a pandas DataFrame.

    Args:
        data_url (str): Path or URL to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        Exception: For any other unexpected error during loading.
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

    Drops unnecessary columns and renames the relevant ones to 
    standardized names for downstream processing.

    Args:
        df (pd.DataFrame): The raw input dataset.

    Returns:
        pd.DataFrame: The cleaned and renamed dataset.

    Raises:
        KeyError: If expected columns are missing from the dataframe.
        Exception: For any other unexpected error during preprocessing.
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

    Args:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The testing dataset.
        data_path (str): Root directory where the 'raw' folder will be created.

    Raises:
        Exception: If saving the datasets fails for any reason.
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


# Main Pipeline Execution


def main():
    """
    Orchestrates the full data ingestion pipeline:
    load -> preprocess -> split -> save.
    """
    try:
        test_size = 0.2

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


    


