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