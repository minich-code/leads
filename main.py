from src.LeadGen.logger import logger
from src.LeadGen.pipelines.pip_01_data_ingestion import DataIngestionPipeline


COMPONENT_01_NAME = "DATA INGESTION COMPONENT"
try:
    logger.info(f"# ====================== {COMPONENT_01_NAME} Started! ============================== #")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.run()
    logger.info(f"# ====================== {COMPONENT_01_NAME} Terminated Successfully! ===============##\n\nx******************x")

except Exception as e:
    logger.exception(e)
    raise e