"""
model_evaluation.py

This module handles the model evaluation stage of the ML pipeline:
    1. Loading the trained model and the TF-IDF transformed test data
    2. Generating predictions and evaluating model performance
    3. Saving the evaluation metrics (accuracy, precision, recall, AUC) 
       to a JSON report

Logging is configured to track each step and capture errors both 
on the console and in a persistent log file.
"""

import os
import json
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# Logging Configuration

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(LOG_DIR, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Model Evaluation Functions

def load_model(file_path: str):
    """
    Load a trained model from a pickle file.

    :param file_path: Path to the saved model file.
    :return: The deserialized model object.
    """
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: Loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate a trained classifier on test data and compute performance metrics.

    :param clf: Trained classifier with predict/predict_proba methods.
    :param x_test: Test feature matrix.
    :param y_test: True labels for the test set.
    :return: Dictionary containing accuracy, precision, recall, and AUC scores.
    """
    try:
        y_pred = clf.predict(x_test)
        # predict_proba returns probabilities for both classes;
        # we only need the probability of the positive class (index 1)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file, creating the target 
    directory if it doesn't exist.

    :param metrics: Dictionary of evaluation metrics.
    :param file_path: Destination path for the JSON file.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise


# Main Pipeline Execution

def main():
    """
    Orchestrates the full model evaluation pipeline:
    load model & test data -> evaluate -> save metrics report.
    """
    try:
        clf = load_model('./models/model.pkl')
        test_data = load_data('./data/processed/test_tfidf.csv')

        x_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, x_test, y_test)

        save_metrics(metrics, 'reports/metrics.json')

    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()

