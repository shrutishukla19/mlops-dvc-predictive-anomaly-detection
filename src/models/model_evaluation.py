import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yaml
import json
import pickle

logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
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
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("Data file not found: %s", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", str(e))
        raise
    

def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        logger.error("Model file not found: %s", model_path)
        raise
    except pickle.PickleError as e:
        logger.error("Error parsing pickle file: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while loading model: %s", str(e))
        raise

def evaluate_model(model, X_test: pd.DataFrame, y_test) -> dict:
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        return metrics_dict
    
    except Exception as e:
        logger.error("Error occurred during model evaluation: %s", str(e))
        raise
   

def save_metrics(metrics: dict, output_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved successfully at: %s", output_path)
    except Exception as e:
        logger.error("Error occurred while saving metrics: %s", str(e))
        raise
   



def main():
    try:
        params = load_params("params.yaml")['model_evaluation']

        test_data = load_data(params.get("test_data_path", "data/processed/test_featured.csv"))

        X_test = test_data.drop(columns=['Machine failure'])
        y_test = test_data['Machine failure']

        model_path = params.get("model_path", "models/model.pkl")
        model = load_model(model_path)

        metrics_dict = evaluate_model(model, X_test, y_test)
        metrics_output_path = params.get("metrics_output_path", "reports/metrics.json")
        
        save_metrics(metrics_dict, metrics_output_path)
    except Exception as e:
        logger.error("An unexpected error occurred in main: %s", str(e))
        raise
    

if __name__ == "__main__":
    main()