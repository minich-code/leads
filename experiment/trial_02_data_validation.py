from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from src.LeadGen.constants import *
from src.LeadGen.utils.commons import read_yaml, create_directories
from src.LeadGen.logger import logger

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict
    critical_columns: list  
    data_ranges: dict

class ConfigurationManager:
    def __init__(
        self, 
        data_validation_config=DATA_VALIDATION_CONFIG_FILEPATH, 
        schema_config=SCHEMA_CONFIG_FILEPATH):
        
        logger.info("Initializing ConfigurationManager")
        self.data_val_config = read_yaml(data_validation_config)
        self.schema = read_yaml(schema_config)
        create_directories([self.data_val_config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        data_val_config = self.data_val_config.data_validation
        schema = self.schema.COLUMNS
        create_directories([data_val_config.root_dir])
        logger.debug("Data validation configuration loaded")
        return DataValidationConfig(
            root_dir=data_val_config.root_dir,
            STATUS_FILE=data_val_config.STATUS_FILE,
            data_dir=data_val_config.data_dir,
            all_schema=schema,
            critical_columns=data_val_config.critical_columns,
            data_ranges=data_val_config.data_ranges
        )

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        logger.info("DataValidation initialized with config")

    def _validate_columns(self, data):
        """Validates if all expected columns are present and no unexpected columns exist."""
        all_cols = list(data.columns)
        all_schema = list(self.config.all_schema.keys())

        missing_columns = [col for col in all_schema if col not in all_cols]
        extra_columns = [col for col in all_cols if col not in all_schema]

        if missing_columns or extra_columns:
            logger.debug(f"Missing columns: {missing_columns}")
            logger.debug(f"Extra columns: {extra_columns}")
            return False
        return True

    def _validate_data_types(self, data):
        """Validates if the data types of each column match the expected schema."""
        all_schema = self.config.all_schema
        type_mismatches = {}
        for col, expected_type in all_schema.items():
            if col in data.columns:
                actual_type = data[col].dtype
                if actual_type != expected_type:
                    type_mismatches[col] = (expected_type, actual_type)
        if type_mismatches:
            logger.debug(f"Type mismatches: {type_mismatches}")
            return False
        return True

    def _validate_missing_values(self, data):
        """Validates if critical columns have any missing values."""
        missing_values = {}
        for col in self.config.critical_columns:
            if data[col].isnull().sum() > 0:
                missing_values[col] = data[col].isnull().sum()
        if missing_values:
            logger.debug(f"Missing values: {missing_values}")
            return False
        return True

    def _validate_data_ranges(self, data):
        """Validates if data values fall within the specified ranges."""
        range_errors = {}
        for col, range_info in self.config.data_ranges.items():
            if col in data.columns:
                if range_info["min"] is not None and data[col].min() < range_info["min"]:
                    range_errors[col] = f"Minimum value ({data[col].min()}) is less than expected minimum ({range_info['min']})"
                if range_info["max"] is not None and data[col].max() > range_info["max"]:
                    range_errors[col] = f"Maximum value ({data[col].max()}) is greater than expected maximum ({range_info['max']})"
        if range_errors:
            logger.debug(f"Range errors: {range_errors}")
            return False, range_errors
        return True, None

    def validate_data(self, data):
        """Performs all data validation checks and returns the overall validation status."""
        validation_results = {}
        validation_results["validate_all_columns"] = self._validate_columns(data)
        validation_results["validate_data_types"] = self._validate_data_types(data)
        validation_results["validate_missing_values"] = self._validate_missing_values(data)
        validation_results["validate_data_ranges"], range_errors = self._validate_data_ranges(data)

        if range_errors:
            validation_results["range_errors"] = range_errors

        with open(self.config.STATUS_FILE, 'w') as f:
            json.dump(validation_results, f, indent=4)

        overall_validation_status = all(validation_results.values())
        return overall_validation_status

if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data = pd.read_csv(data_validation_config.data_dir)

        logger.info("Starting data validation process")
        validation_status = data_validation.validate_data(data)

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise