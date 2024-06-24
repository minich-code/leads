from src.LeadGen.logger import logger
from src.LeadGen.entity.config_entity import DataValidationConfig


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