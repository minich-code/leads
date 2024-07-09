from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_05_model_evaluation import ModelEvaluation

from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from pathlib import Path 
import json 
import mlflow


PIPELINE_NAME = "MODEL EVALUATION PIPELINE"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def run(self):
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(model_eval_config)
        with mlflow.start_run():
            model_evaluator.evaluate()


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        model_evaluation_pipeline = ModelEvaluationPipeline()
        model_evaluation_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=======++++==========\n\nx************************x")
    except Exception as e:
        logger.error(f"Model Trainer failed: {e}")
        raise