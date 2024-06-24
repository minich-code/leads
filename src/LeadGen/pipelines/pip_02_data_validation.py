from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_02_data_validation import DataValidation
from src.LeadGen.logger import logger 
import pandas as pd 
import json 


PIPELINE_NAME = "DATA VALIDATION PIPELINE"

class DataValidationPipeline:
    
    def __init__(self):
        pass 


    def run(self):
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


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_validation_pipeline = DataValidationPipeline()
        data_validation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise