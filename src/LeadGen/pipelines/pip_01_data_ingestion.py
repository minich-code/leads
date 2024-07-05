from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_01_data_ingestion import DataIngestion

PIPELINE_NAME = "DATA INGESTION PIPELINE"

class DataIngestionPipeline: 
    
    def __init__(self):
        pass 

    def run(self):
        try:
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.import_data_from_mongodb()
            logger.info("Data Ingestion from MongoDB Completed!")
        except CustomException as e:
            logger.error(f"Data ingestion process failed: {e}")


if __name__ == "__main__":
    try:
        logger.info(f"## =================== Starting {PIPELINE_NAME} pipeline ========================##")
        data_ingestion_pipeline = DataIngestionPipeline()
        data_ingestion_pipeline.run()
        logger.info(f"## =============== {PIPELINE_NAME} Terminated Successfully!=================\n\nx************************x")
    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        raise e
