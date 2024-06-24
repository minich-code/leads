from dataclasses import dataclass
from pathlib import Path


# Data Ingestion entity 
@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int


# Data Validation 
@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    data_dir: Path
    all_schema: dict
    critical_columns: list  
    data_ranges: dict