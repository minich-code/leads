from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_05_model_validation import ModelValidation
from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from pathlib import Path 
import json 
import mlflow


PIPELINE_NAME = "MODEL VALIDATION PIPELINE"

class ModelValidationPipeline:
    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_validation_config = config_manager.get_model_validation_config()
        model_validator = ModelValidation(config=model_validation_config)
        model_validator.validate()


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        model_validation_pipeline = ModelValidationPipeline()
        model_validation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=======++++==========\n\nx************************x")
    except Exception as e:
        logger.error(f"Model Validation failed: {e}")
        raise