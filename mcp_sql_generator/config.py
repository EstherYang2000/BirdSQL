# mcp_sql_generator/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
DEFAULT_OPENAI_MODEL = "gpt-4o" # Use a strong default
DEFAULT_GEMINI_MODEL = "models/gemini-2.5-pro-preview-03-25" # Use a strong default
DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free" # Use a strong default

# --- Ensemble Expert Definitions ---
# List of dictionaries, each defining an expert configuration
# We will use these names in WMA
ENSEMBLE_EXPERTS = [
    # {"name": "gpt-4o", "provider": "openai", "model": "chatgpt-4o-latest"},
    # {"name": "gemini-2.5-pro-preview-03-25", "provider": "gemini", "model": "models/gemini-2.5-pro-preview-03-25"},
    {"name": "llama-3.3-70b", "provider": "togetherai", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"},
    # Add more experts if desired, e.g.:
    # {"name": "mixtral-8x7b", "provider": "togetherai", "model": "mistralai/Mixtral-8x7B-Instruct-v0.1"},
    # {"name": "gpt-3.5-turbo", "provider": "openai", "model": "gpt-3.5-turbo"},
]

# --- Agent Configuration ---
DEFAULT_MAX_REFINEMENTS = 5 # Set to 0 if refinement happens AFTER ensemble selection
REQUEST_DELAY_SECONDS = 1.0 # Adjust as needed

# --- File Paths ---
DEFAULT_QUESTION_FILE = 'bird/bird/dev.json'
DEFAULT_SCHEMA_FILE = 'bird/bird/tables.json'
# Output file name for ensemble results
DEFAULT_OUTPUT_FILE = 'bird/output/ensemble_predictions.jsonl'

# --- Processing Configuration ---
DEFAULT_MAX_ITEMS = 1534
DEFAULT_START_INDEX = 0

# --- WMA Configuration ---
DEFAULT_WMA_STRATEGY = "wma" # 'wma', 'rwma', 'naive'
DEFAULT_WMA_EPSILON = 0.005 # Default epsilon if not auto-calculated
DEFAULT_WMA_AUTO_EPSILON = False # Whether to use auto-epsilon calculation

# --- Evaluation Configuration ---
# Define paths needed by evaluate_cc and build_foreign_key_map_from_json
# These might vary based on dev/test and spider/bird
DEFAULT_TABLES_FILE_DEV = "bird/bird/tables.json" # Adjust as per your structure
DEFAULT_TABLES_FILE_TEST = "bird/bird/tables.json" # Adjust (if test tables file exists)
DEFAULT_DB_DIR_DEV = "bird/bird/database/"        # Adjust
DEFAULT_DB_DIR_TEST = "bird/bird/database/"       # Adjust (if test DB dir exists)

DEFAULT_EVALUATION_TIMEOUT = 30.0


DEFAULT_FEW_SHOT_TRAIN_DATA_PATH = "bird/bird/train.json" # CORRECT PATH
DEFAULT_FEW_SHOT_TABLES_PATH = "bird/bird/tables.json"    # CORRECT PATH
# Paths for files you DON'T have - set to None or remove if selector logic handles absence
DEFAULT_TRAIN_SCHEMA_LINKING_PATH = None # Or path if you generate it
DEFAULT_TEST_SCHEMA_LINKING_PATH = None  # Or path if you generate it
DEFAULT_FEW_SHOT_RESULTS_PATH = None   # Or path if you create it