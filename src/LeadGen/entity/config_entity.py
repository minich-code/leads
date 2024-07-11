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


# Data Preprocessing
@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    numerical_cols: list
    categorical_cols: list

# Model Trainer 
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_name: str
    train_features_path: Path
    train_targets_path: Path
    val_features_path: Path
    val_targets_path: Path
    val_metrics_path: Path
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict  
    optimizer: str
    loss_function: str
    activation_function: str

# Model evaluation 
@dataclass()
class ModelEvaluationConfig:
    root_dir: Path
    val_features_path: Path
    val_target_path: Path
    model_path: Path
    metric_file_name: Path
    validation_metrics_path: Path
    training_metrics_path: Path
    report_path: Path
    confusion_matrix_report: Path
    # Model parameters
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict
    optimizer: str
    loss_function: str
    activation_function: str


@dataclass
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_target_path: Path
    model_path: Path
    class_report: Path
    #classification_report_path: Path
    val_metric_file_name: Path
    #conf_matrix: Path
    #roc_auc_path: Path
    #pr_auc_path: Path

    # Model parameters
    batch_size: int
    learning_rate: float
    epochs: int
    dropout_rates: dict
    optimizer: str
    loss_function: str
    activation_function: str



