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
        validation_status = data_validation.validate_data(data)

        if validation_status:
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