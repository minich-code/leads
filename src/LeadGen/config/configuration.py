from pathlib import Path 
from src.LeadGen.utils.commons import read_yaml, create_directories
from src.LeadGen.constants import *

from src.LeadGen.entity.config_entity import (DataIngestionConfig, DataValidationConfig)
from src.LeadGen.logger import logger      

# ConfigurationManager class to manage configurations
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH, schema_filepath=SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

# Data ingestion Config
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        return DataIngestionConfig(
            root_dir=config.root_dir,
            mongo_uri=config.mongo_uri,
            database_name=config.database_name,
            collection_name=config.collection_name,
            batch_size=config.get('batch_size', 3000)
        )

# Data validation Config 
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