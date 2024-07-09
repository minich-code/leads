from src.LeadGen.config.configuration import ConfigurationManager
from src.LeadGen.components.c_03_data_transformation import DataTransformation
from src.LeadGen.logger import logger 
from src.LeadGen.exception import CustomException
from pathlib import Path 
import json 


PIPELINE_NAME = "DATA TRANSFORMATION PIPELINE"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def run(self):
        try:
            # Read the JSON file
            with open(Path("artifacts/data_validation/status.json"), "r") as f:
                validation_results = json.load(f)

            # Check if all validation statuses are true
            if (
                validation_results["validate_all_columns"]
                and validation_results["validate_data_types"]
                and validation_results["validate_missing_values"]
                and validation_results["validate_data_ranges"]
            ):
                logger.info(
                    f"The data validation pipeline has already been executed Successfully !!!!"
                )

                logger.info(f"#====================== {PIPELINE_NAME} Started ================================#")
            
                config_manager = ConfigurationManager()
                data_transformation_config = config_manager.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor = data_transformation.train_val_test_split()
                X_train_tensor, X_val_tensor, X_test_tensor, preprocessor_path = data_transformation.initiate_data_transformation(
                    X_train, X_val, X_test, y_train_tensor, y_val_tensor, y_test_tensor)
                
                # Store data 
                #data_transformation.save_data(X_train_tensor, X_val_tensor, X_test_tensor,  y_train_tensor, y_val_tensor, y_test_tensor, preprocessor_path)


                logger.info("Data Transformation process has completed")
                logger.info(f"#======================== {PIPELINE_NAME} Terminated Successfully ! ===========\n\nx*************x")

            else:
                raise Exception("The Data Schema is Invalid")

            
        except Exception as e:
            logger.exception("Failed during data transformation.")
            raise CustomException(f"Error during data transformation: {e}")
        

if __name__ == "__main__":
    try:
        logger.info(f"# ============== {PIPELINE_NAME} Started ================#")
        data_transformation_pipeline = DataTransformationPipeline()
        data_transformation_pipeline.run()
        logger.info(f"# ============= {PIPELINE_NAME} Terminated Successfully! ======+++=====\n\nx*********************x") 
    except Exception as e: 
        logger.exception("Failed during data transformation.")
        raise CustomException(f"Error during data transformation: {e}")
