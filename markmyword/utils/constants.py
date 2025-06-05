import configparser
import os
parser = configparser.ConfigParser()
parser.read(os.path.join(os.path.dirname(__file__),'../config/config.conf'))
#API ACCESS
AZURE_API_KEY = parser.get('AZURE','api')
AZURE_ENDPOINT = parser.get('AZURE','endpoint')
AZURE_DEPLOYMENT_NAME = parser.get('AZURE','model_name')
AZURE_API_VERSION = parser.get('AZURE','version')
#DATA PATHS
# Get the project root (one level above the script directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_BOOK_DIR = os.path.join(PROJECT_ROOT, "data", "raw_book")
CLEANED_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "transform_book")
ENRICHED_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "final")