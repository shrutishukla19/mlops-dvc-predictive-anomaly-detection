import logging
import os
from typing import Any

import pandas as pd
import yaml


logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str = "params.yaml") -> dict[str, Any]:
	try:
		with open(params_path, "r", encoding="utf-8") as file:
			params = yaml.safe_load(file) or {}
		return params.get("feature_engineering", {})
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


def create_features(data: pd.DataFrame) -> pd.DataFrame:
	required_columns = [
		"Process temperature [K]",
		"Air temperature [K]",
		"Torque [Nm]",
		"Rotational speed [rpm]",
		"Tool wear [min]",
	]
	missing_columns = [
		column for column in required_columns if column not in data.columns
	]
	if missing_columns:
		raise ValueError(
			"Cannot create features; missing columns: "
			+ ", ".join(missing_columns)
		)

	featured_data = data.copy()
	featured_data["Temperature_Difference"] = (
		featured_data["Process temperature [K]"]
		- featured_data["Air temperature [K]"]
	)
	featured_data["Mechanical_Load"] = (
		featured_data["Torque [Nm]"]
		* featured_data["Rotational speed [rpm]"]
	)
	featured_data["Wear_Stress"] = (
		featured_data["Tool wear [min]"] * featured_data["Torque [Nm]"]
	)
	featured_data["Wear_Heat"] = (
		featured_data["Tool wear [min]"]
		* featured_data["Process temperature [K]"]
	)
	return featured_data


def save_data(
	train_data: pd.DataFrame,
	test_data: pd.DataFrame,
	output_dir: str,
	train_filename: str = "train_featured.csv",
	test_filename: str = "test_featured.csv",
) -> None:
	try:
		os.makedirs(output_dir, exist_ok=True)
		train_data.to_csv(os.path.join(output_dir, train_filename), index=False)
		test_data.to_csv(os.path.join(output_dir, test_filename), index=False)
		logger.debug("Featured data saved to %s", output_dir)
	except Exception as e:
		logger.error("Error saving featured data: %s", str(e))
		raise


def main() -> None:
	try:
		params = load_params()
		input_dir = params.get("input_dir", "data/interim")
		output_dir = params.get("output_dir", "data/processed")
		train_filename = params.get("train_filename", "train_processed.csv")
		test_filename = params.get("test_filename", "test_processed.csv")

		train_data = create_features(
			load_data(os.path.join(input_dir, train_filename))
		)
		test_data = create_features(load_data(os.path.join(input_dir, test_filename)))
		save_data(train_data, test_data, output_dir)
		logger.info("Featured data saved to %s", output_dir)
	except Exception as e:
		logger.error("An unexpected error occurred in main: %s", str(e))
		raise
	


if __name__ == "__main__":
	main()
