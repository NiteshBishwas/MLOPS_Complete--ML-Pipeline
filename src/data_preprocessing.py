"""
data_preprocessing.py

This module handles the data preprocessing stage of the ML pipeline:
    1. Cleaning and normalizing text data (lowercase, tokenize, 
       remove stopwords/punctuation, stemming)
    2. Encoding the target column into numeric labels
    3. Removing duplicate rows
    4. Saving the processed train/test datasets for the next stage

Logging is configured to track each step of preprocessing and to 
capture errors both on the console and in a persistent log file.
"""

import os
import string
import logging

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


# NLTK Setup


# Download required NLTK data packages 
nltk.download('stopwords')
nltk.download('punkt')

# Logging Configuration


# Ensure the "logs" directory exists to store log files
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Create a logger specific to the data preprocessing module
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Handler to print logs to the console (useful during development)
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# Handler to persist logs to a file (useful for later debugging/auditing)
log_file_path = os.path.join(LOG_DIR, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Define a consistent log message format: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach both handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# Text Preprocessing Functions


def transform_text(text: str) -> str:
    """
    Clean and normalize a single piece of text.

    Steps performed:
        1. Convert text to lowercase
        2. Tokenize into individual words
        3. Keep only alphanumeric tokens (remove special characters)
        4. Remove stopwords and punctuation
        5. Apply stemming to reduce words to their root form

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and normalized text.
    """
    ps = PorterStemmer()

    # Step 1: Lowercase the text for uniformity
    text = text.lower()

    # Step 2: Break the text into individual word tokens
    text = nltk.word_tokenize(text)

    # Step 3: Keep only alphanumeric tokens (drop punctuation-only tokens)
    text = [word for word in text if word.isalnum()]

    # Step 4: Remove common stopwords and any remaining punctuation
    text = [
        word for word in text
        if word not in stopwords.words('english') and word not in string.punctuation
    ]

    # Step 5: Reduce each word to its root/stem form
    text = [ps.stem(word) for word in text]

    # Join the cleaned tokens back into a single string
    return " ".join(text)


def preprocess_df(df: pd.DataFrame, text_column: str = 'text', target_column: str = 'target') -> pd.DataFrame:
    """
    Preprocess a DataFrame by encoding the target column, removing 
    duplicate rows, and cleaning the text column.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        text_column (str): Name of the column containing raw text.
        target_column (str): Name of the column containing labels.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    Raises:
        KeyError: If the specified columns are not found in the DataFrame.
        Exception: For any other unexpected error during preprocessing.
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')

        # Encode the target column into numeric labels (e.g., spam/ham -> 1/0)
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows, keeping the first occurrence
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        # Apply text cleaning to every row in the text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')

        return df

    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise



# Main Pipeline Execution


def main(text_column: str = 'text', target_column: str = 'target'):
    """
    Orchestrates the full preprocessing pipeline:
    load raw data -> preprocess -> save processed data.
    """
    try:
        # Load the raw train and test datasets produced by data ingestion
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Preprocess both datasets using the same logic
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Ensure the "interim" directory exists to store processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        # Save the processed datasets as CSV files
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logger.debug('Processed data saved to %s', data_path)

    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
