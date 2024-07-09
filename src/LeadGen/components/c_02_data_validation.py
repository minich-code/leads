import json 
from src.LeadGen.logger import logger
from src.LeadGen.entity.config_entity import DataValidationConfig


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