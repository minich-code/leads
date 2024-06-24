from dataclasses import dataclass
from pathlib import Path
import os
import json
import pandas as pd

from src.LeadGen.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
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
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):
        
        logger.info("Initializing ConfigurationManager")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])
        logger.debug("Data validation configuration loaded")
        return DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir=config.data_dir,
            all_schema=schema,
            critical_columns=config.critical_columns,
            data_ranges=config.data_ranges
        )

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        logger.info("DataValidation initialized with config")

    def validate_all_columns(self, data):
        try:
            validation_status = True
            all_cols = list(data.columns)
            all_schema = list(self.config.all_schema.keys())

            missing_columns = [col for col in all_schema if col not in all_cols]
            extra_columns = [col for col in all_cols if col not in all_schema]

            if missing_columns or extra_columns:
                validation_status = False
                logger.debug(f"Missing columns: {missing_columns}")
                logger.debug(f"Extra columns: {extra_columns}")

            logger.info(f"All columns validation status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error(f"Error in validate_all_columns: {e}")
            raise e

    def validate_data_types(self, data):
        try:
            validation_status = True
            all_schema = self.config.all_schema

            type_mismatches = {}
            for col, expected_type in all_schema.items():
                if col in data.columns:
                    actual_type = data[col].dtype
                    if actual_type != expected_type:
                        type_mismatches[col] = (expected_type, actual_type)
                        validation_status = False

            if type_mismatches:
                logger.debug(f"Type mismatches: {type_mismatches}")

            logger.info(f"Data types validation status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error(f"Error in validate_data_types: {e}")
            raise e

    def validate_missing_values(self, data):
        try:
            validation_status = True
            missing_values = {}

            for col in self.config.critical_columns:
                if data[col].isnull().sum() > 0:
                    missing_values[col] = data[col].isnull().sum()
                    validation_status = False

            if missing_values:
                logger.debug(f"Missing values: {missing_values}")

            logger.info(f"Missing values validation status: {validation_status}")
            return validation_status

        except Exception as e:
            logger.error(f"Error in validate_missing_values: {e}")
            raise e
        
    def validate_data_ranges(self, data):
        try:
            validation_status = True
            range_errors = {}
            for col, range_info in self.config.data_ranges.items():
                if col in data.columns:
                    if range_info["min"] is not None and data[col].min() < range_info["min"]:
                        range_errors[col] = f"Minimum value ({data[col].min()}) is less than expected minimum ({range_info['min']})"
                        validation_status = False
                    if range_info["max"] is not None and data[col].max() > range_info["max"]:
                        range_errors[col] = f"Maximum value ({data[col].max()}) is greater than expected maximum ({range_info['max']})"
                        validation_status = False

            if range_errors:
                logger.debug(f"Range errors: {range_errors}")
                return validation_status, range_errors
            else:
                logger.info("Data ranges validation passed")
                return validation_status, None

        except Exception as e:
            logger.error(f"Error in validate_data_ranges: {e}")
            raise e

if __name__ == "__main__":
    try:
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data = pd.read_csv(data_validation_config.data_dir)
        
        logger.info("Starting data validation process")
        
        column_validation_status = data_validation.validate_all_columns(data)
        type_validation_status = data_validation.validate_data_types(data)
        missing_values_status = data_validation.validate_missing_values(data)
        range_validation_status, range_errors = data_validation.validate_data_ranges(data)

        validation_results = {
            "validate_all_columns": column_validation_status,
            "validate_data_types": type_validation_status,
            "validate_missing_values": missing_values_status,
            "validate_data_ranges": range_validation_status
        }

        if range_errors:
            validation_results["range_errors"] = range_errors

        with open(data_validation_config.STATUS_FILE, 'w') as f:
            json.dump(validation_results, f, indent=4)

        overall_validation_status = (
            column_validation_status and
            type_validation_status and
            missing_values_status and
            range_validation_status
        )

        if overall_validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")
    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise
