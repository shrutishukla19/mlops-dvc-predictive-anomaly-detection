import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import yaml

import os

import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        logger.debug(f'test_size {test_size} retrieved')
        return test_size
    except FileNotFoundError:
        logger.error('File not found error')
        raise
    except yaml.YAMLError as e:
        logger.error(f'yaml error: {e}')
        raise
    except Exception as e:
        logger.error(f'Error occured: {e}')
        raise

def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(f'Failed to parse the csv file from {data_url}: {e}')
        raise
    except Exception as e:
        logger.error(f'An unexpected error occurred while loading the data: {e}')
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
    except Exception as e:
        logger.error(f'An unexpected error occurred while saving the data: {e}')
        raise


def main():
    try:
        test_size = load_params(params_path="params.yaml")
        df = load_data("data/external/ai4i2020.csv")
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        save_data(train_data=train_data, test_data=test_data, data_path="data")
    except Exception as e:
        logger.error(f'An unexpected error occurred: {e}')
        raise


if __name__ == "__main__":
    main()




