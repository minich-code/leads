import logging
import logging.config
import os
import sys
from datetime import datetime

# Define the logfile name using the current date and time
log_file_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Create the path to the log file
log_file_path = os.path.join(os.getcwd(), "logs", log_file_name)

# Create the log directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Logger configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',  # Log all levels to the file
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'filename': log_file_path,
            'when': 'midnight',
            'backupCount': 7,
            'formatter': 'detailed',
        },
        'console': {
            'level': 'INFO',  # Log info and higher levels to the console
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'standard',
        },
    },
    'loggers': {
        'Leads': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',  # Set the minimum logging level to DEBUG
            'propagate': False
        }
    }
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Get the logger
logger = logging.getLogger('Leads')

# # Example usage of different logging levels
# def example_function():
#     logger.debug("This is a debug message for detailed diagnostic purposes.")
#     logger.info("This is an info message to indicate a general event.")
#     try:
#         # Simulate an error
#         1 / 0
#     except ZeroDivisionError as e:
#         logger.error("This is an error message to indicate an issue.", exc_info=True)

# if __name__ == "__main__":
#     logger.info("Starting the lead scoring process...")
#     example_function()
#     logger.info("Finished the lead scoring process.")
