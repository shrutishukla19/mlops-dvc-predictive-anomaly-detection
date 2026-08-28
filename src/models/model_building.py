import pandas as pd
import os
import logging

from sklearn.ensemble import RandomForestClassifier
import yaml
import pickle

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        logger.error("Parameter file not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML file: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while loading parameters: %s", str(e))
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", str(e))
        raise
    

def train_model(X_train: pd.DataFrame, y_train, params: dict) -> RandomForestClassifier:
    try:
        rf = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            random_state=42,
            class_weight=params.get("class_weight", "balanced")
        )
        rf.fit(X_train, y_train)
        logger.debug("Model trained successfully with parameters: %s", params)
        return rf
    except Exception as e:
        logger.error("Error occurred while training the model: %s", str(e))
        raise
  

def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug("Model saved successfully at: %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving the model: %s", str(e))
        raise
   

def main():
    try:
        params = load_params("params.yaml")['model_building']

        input_dir = params.get("input_dir", "data/processed")
        output_dir = params.get("output_dir", "models")

        train_filename = params.get("train_filename", "train_featured.csv")

        train_data = load_data(os.path.join(input_dir, train_filename))

        X_train = train_data.drop(columns=['Machine failure'])
        y_train = train_data['Machine failure']
        

        model = train_model(X_train, y_train, params)

        save_model(model, params.get("model_output_path", "models/model.pkl"))
    except Exception as e:
        logger.error("An unexpected error occurred in main: %s", str(e))
        raise

if __name__ == "__main__":
    main()

