"""
feature_engineering.py

This module handles the feature engineering stage of the ML pipeline:
    1. Loading the preprocessed train/test data
    2. Converting text into numerical features using TF-IDF
    3. Saving the transformed features for model training

Logging is configured to track each step and capture errors both 
on the console and in a persistent log file.
"""

import os
import logging

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# Logging Configuration


LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(LOG_DIR, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Feature Engineering Functions


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a processed dataset from a CSV file and fill missing values.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset with NaNs filled as empty strings.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        Exception: For any other unexpected error during loading.
    """
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Convert text data into numerical features using TF-IDF.

    Args:
        train_data (pd.DataFrame): Preprocessed training data with 'text' and 'target' columns.
        test_data (pd.DataFrame): Preprocessed testing data with 'text' and 'target' columns.
        max_features (int): Maximum number of TF-IDF features to keep.

    Returns:
        tuple: (train_df, test_df) — DataFrames with TF-IDF features and a 'label' column.

    Raises:
        Exception: If TF-IDF transformation fails for any reason.
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        x_train = train_data['text'].values
        y_train = train_data['target'].values
        x_test = test_data['text'].values
        y_test = test_data['target'].values

        # Learn vocabulary from training data and transform it
        x_train_bow = vectorizer.fit_transform(x_train)
        # Use the same learned vocabulary to transform test data
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug("TF-IDF applied and data transformed")
        return train_df, test_df

    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file, creating the target directory if needed.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): Destination path for the CSV file.

    Raises:
        Exception: If saving the file fails for any reason.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


# Main Pipeline Execution


def main():
    """
    Orchestrates the full feature engineering pipeline:
    load processed data -> apply TF-IDF -> save transformed features.
    """
    try:
        max_features = 50

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))

    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()



      
        
        