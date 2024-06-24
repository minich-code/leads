import time 
from datetime import datetime 
import pandas as pd 
import pymongo 
import os 
import json

from src.LeadGen.exception import CustomException
from src.LeadGen.logger import logger
from src.LeadGen.entity.config_entity import DataIngestionConfig


# DataIngestion component class
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(self.config.mongo_uri)
            db = client[self.config.database_name]
            collection = db[self.config.collection_name]
            logger.info(f"Starting data ingestion from MongoDB collection {self.config.collection_name}")

            batch_size = self.config.batch_size
            batch_num = 0
            all_data = pd.DataFrame()

            while True:
                cursor = collection.find().skip(batch_num * batch_size).limit(batch_size)
                df = pd.DataFrame(list(cursor))

                if df.empty:
                    break

                if "_id" in df.columns:
                    df = df.drop(columns=["_id"])

                all_data = pd.concat([all_data, df], ignore_index=True)
                logger.info(f"Fetched batch {batch_num + 1} with {len(df)} records")
                batch_num += 1

            output_path = os.path.join(self.config.root_dir, 'lead.csv')
            all_data.to_csv(output_path, index=False)
            logger.info(f"Data fetched from MongoDB and saved to {output_path}")

            total_records = len(all_data)
            logger.info(f"Total records ingested: {total_records}")

            self._save_metadata(start_timestamp, start_time, total_records, output_path)

        except pymongo.errors.PyMongoError as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            raise CustomException(f"Error connecting to MongoDB: {e}")
        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise CustomException(f"Error during data ingestion: {e}")

    def _save_metadata(self, start_timestamp: str, start_time: float, total_records: int, output_path: str):
        end_time = time.time()
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        duration = end_time - start_time
        metadata = {
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'duration': duration,
            'total_records': total_records,
            'data_source': self.config.collection_name,
            'output_path': output_path
        }
        metadata_path = os.path.join(self.config.root_dir, 'data-ingestion-metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")