from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_04_model_trainer import ModelTrainer

from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from pathlib import Path 
import json 


PIPELINE_NAME = "MODEL TRAINER PIPELINE"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train_model()


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        model_trainer_pipeline = ModelTrainerPipeline()
        model_trainer_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Model Trainer failed: {e}")
        raise