import logging
import os
from typing import Any

import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder


logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('data_preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_params(params_path: str = 'params.yaml') -> dict:
		try:
			with open(params_path, 'r') as file:
				params = yaml.safe_load(file)
			logger.debug(f'Parameters loaded from {params_path}')
			return params.get('data_preprocessing', {})
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
		logger.error("Error occurred while loading data from %s: %s", data_path, str(e))
		raise

def drop_columns(
	df: pd.DataFrame, columns_to_drop: list[str]
) -> pd.DataFrame:
	try:
		return df.drop(columns=columns_to_drop, errors="ignore")
	except KeyError as e:
		logger.error("Error dropping columns: %s", str(e))
		raise
	except Exception as e:
		logger.error("Unexpected error while dropping columns: %s", str(e))
		raise



def encoding(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	try:
		categorical_columns = train_df.select_dtypes(
			include=["object", "string", "category", "bool"]
		).columns.tolist()

		if not categorical_columns:
			return train_df.copy(), test_df.copy()

		encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
		train_encoded = encoder.fit_transform(train_df[categorical_columns])
		test_encoded = encoder.transform(test_df[categorical_columns])
		encoded_columns = encoder.get_feature_names_out(categorical_columns)

		train_numeric = train_df.drop(columns=categorical_columns)
		test_numeric = test_df.drop(columns=categorical_columns)
		train_encoded_df = pd.DataFrame(
			train_encoded, columns=encoded_columns, index=train_df.index
		)
		test_encoded_df = pd.DataFrame(
			test_encoded, columns=encoded_columns, index=test_df.index
		)

		return (
			pd.concat([train_numeric, train_encoded_df], axis=1),
			pd.concat([test_numeric, test_encoded_df], axis=1),
		)
	except Exception as e:
		logger.error("Error during encoding: %s", str(e))
		raise


def save_data(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	output_dir: str,
) -> None:
	try:
		os.makedirs(output_dir, exist_ok=True)
		train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
		test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
		logger.debug("Processed data saved to %s", output_dir)
	except Exception as e:
		logger.error("Error saving processed data: %s", str(e))
		raise


def main() -> None:
	try:
		params = load_params()
		input_dir = params.get("input_dir", "data/raw")
		output_dir = params.get("output_dir", "data/interim")
		columns_to_drop = params.get("columns_to_drop", [])

		train_df = drop_columns(
			load_data(os.path.join(input_dir, "train.csv")), columns_to_drop
		)
		test_df = drop_columns(
			load_data(os.path.join(input_dir, "test.csv")), columns_to_drop
		)
		train_df, test_df = encoding(train_df, test_df)
		save_data(train_df, test_df, output_dir)
		logger.info("Processed data saved to %s", output_dir)
	except Exception as e:
		logger.error("An unexpected error occurred in main: %s", str(e))
		raise

if __name__ == "__main__":
	main()





