# mcp_sql_generator/main_ensemble.py
import argparse
import config
import logging
import os
import sys # For path manipulation if needed

# --- Add project root to Python path if necessary ---
# Ensures modules like evaluation_utils, agents, etc., can be found
# Adjust the number of os.path.dirname based on where you run the script from
# If running from BirdSQL directory:
# project_root = os.path.dirname(os.path.abspath(__file__)) # Gets directory of main_ensemble.py
# mcp_sql_generator_dir = os.path.join(project_root) # Already in the right place if running from parent dir
# Or if running from mcp_sql_generator directory:
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level
sys.path.insert(0, project_root) # Add project root to path
# --- End Path Addition ---

from mcp_sql_generator.llm_interface import get_llm_client, LLMInterface
from mcp_sql_generator.ensemble_orchestrator import EnsembleOrchestrator
from mcp_sql_generator.utils import load_question_data, load_schemas, save_results
from tqdm import tqdm

# --- Logging Setup ---
# Configure root logger (use basicConfig for simplicity, or more advanced setup)
logging.basicConfig(
    level=logging.INFO, # Set default level (e.g., INFO, DEBUG)
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Output to console
)
# Optional: Configure file logging
# log_file = "ensemble_run.log"
# file_handler = logging.FileHandler(log_file, mode='a') # Append mode
# file_handler.setLevel(logging.DEBUG) # Log more details to file
# file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# file_handler.setFormatter(file_formatter)
# logging.getLogger().addHandler(file_handler) # Add handler to root logger

logger = logging.getLogger(__name__) # Get logger for this module

# --- NLTK Download (Optional but recommended) ---
try:
    import nltk
    logger.debug("Checking NLTK data...")
    try: nltk.data.find('tokenizers/punkt')
    except: logger.info("Downloading NLTK 'punkt' data..."); nltk.download('punkt', quiet=True)
    # try: nltk.data.find('corpora/wordnet') # May not be needed by evaluation_utils
    # except: logger.info("Downloading NLTK 'wordnet' data..."); nltk.download('wordnet', quiet=True)
    logger.debug("NLTK data check complete.")
except ImportError:
    logger.warning("NLTK not installed. Text processing in dependencies might fail.")
except Exception as e:
     logger.warning(f"Error during NLTK check/download: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate SQL using an Ensemble Agent System with WMA and BIRD Evaluation.")
    # --- File Paths ---
    parser.add_argument('--question_file', type=str, default=config.DEFAULT_QUESTION_FILE, help="Path to input question JSON file.")
    parser.add_argument('--schema_file', type=str, default=config.DEFAULT_SCHEMA_FILE, help="Path to schema JSON file.")
    parser.add_argument('--output_file', type=str, default=config.DEFAULT_OUTPUT_FILE, help="Path to save output JSON Lines file.")
    # --- Processing Control ---
    parser.add_argument('--start_index', type=int, default=config.DEFAULT_START_INDEX, help="Index of first item to process.")
    parser.add_argument('--max_items', type=int, default=config.DEFAULT_MAX_ITEMS, help="Max items to process (defaults to all).")
    # --- Agent/Refinement Config ---
    parser.add_argument('--max_refinements', type=int, default=config.DEFAULT_MAX_REFINEMENTS, help="Max internal refinement iterations per expert.")
    # --- WMA Config ---
    parser.add_argument('--strategy', type=str, default=config.DEFAULT_WMA_STRATEGY, choices=["wma", "rwma", "naive"], help="WMA voting strategy.")
    parser.add_argument('--epsilon', type=float, default=config.DEFAULT_WMA_EPSILON, help="WMA epsilon value (if not using auto-epsilon).")
    parser.add_argument("--auto_epsilon", action="store_true", default=config.DEFAULT_WMA_AUTO_EPSILON, help="Enable auto-epsilon calculation.")
    # --- Evaluation Config ---
    parser.add_argument('--data_type', type=str, default='bird', choices=['bird', 'spider'], help="Type of dataset (for DB path/eval).")
    parser.add_argument('--dataset_type', type=str, default='dev', choices=['dev', 'test'], help="Dataset split (dev/test).")
    parser.add_argument('--sql_dialect', type=str, default="SQLite", choices=["SQLite", "MySQL", "PostgreSQL"], help="SQL dialect for evaluation connection.")
    parser.add_argument('--eval_timeout', type=float, default=config.DEFAULT_EVALUATION_TIMEOUT, help="Timeout in seconds for SQL execution during evaluation.")

    args = parser.parse_args()

    # --- Log Configuration ---
    logger.info("--- Ensemble System Configuration ---")
    logger.info(f"Question File: {args.question_file}")
    logger.info(f"Schema File: {args.schema_file}")
    logger.info(f"Output File: {args.output_file}")
    logger.info(f"Expert Configurations: {len(config.ENSEMBLE_EXPERTS)} defined")
    for i, exp in enumerate(config.ENSEMBLE_EXPERTS): logger.info(f"  Expert {i+1}: {exp.get('name', 'N/A')} ({exp.get('provider', 'N/A')}/{exp.get('model', 'N/A')})")
    logger.info(f"Max Internal Refinements: {args.max_refinements}")
    logger.info(f"WMA Strategy: {args.strategy}")
    logger.info(f"WMA Auto Epsilon: {args.auto_epsilon}")
    if not args.auto_epsilon: logger.info(f"WMA Manual Epsilon: {args.epsilon}")
    logger.info(f"Max Items: {'All' if args.max_items is None else args.max_items}")
    logger.info(f"Start Index: {args.start_index}")
    logger.info(f"Data Type: {args.data_type}")
    logger.info(f"Dataset Type: {args.dataset_type}")
    logger.info(f"SQL Dialect: {args.sql_dialect}")
    logger.info(f"Evaluation Timeout: {args.eval_timeout}s")
    logger.info("------------------------------------")


    # --- LLM Client Initialization ---
    logger.info("Initializing LLM clients...")
    llm_clients: dict[str, LLMInterface] = {}
    required_providers = sorted(list(set(exp['provider'] for exp in config.ENSEMBLE_EXPERTS))) # Sort for consistent init order
    init_errors = False
    for provider in required_providers:
         try:
             # Find the *first* expert config using this provider to get a model name for init
             # If multiple experts use the same provider but different models, this only uses the first one found.
             # The specific model used by each expert is determined later by the orchestrator using the config.
             model_for_provider_init = next((exp['model'] for exp in config.ENSEMBLE_EXPERTS if exp['provider'] == provider), None)

             # Retrieve default models from config in case the provider isn't explicitly listed with a model
             default_openai = getattr(config, 'DEFAULT_OPENAI_MODEL', 'gpt-4o') # Provide fallbacks
             default_gemini = getattr(config, 'DEFAULT_GEMINI_MODEL', 'gemini-1.5-pro-latest')
             default_together = getattr(config, 'DEFAULT_TOGETHER_MODEL', 'meta-llama/Llama-3-70b-chat-hf')

             # Determine the model to use for initialization - prefer the one from config, fallback to defaults
             openai_model_to_init = model_for_provider_init if provider == 'openai' and model_for_provider_init else default_openai
             gemini_model_to_init = model_for_provider_init if provider == 'gemini' and model_for_provider_init else default_gemini
             together_model_to_init = model_for_provider_init if provider == 'togetherai' and model_for_provider_init else default_together

             llm_clients[provider] = get_llm_client(
                 provider=provider,
                 openai_api_key=config.OPENAI_API_KEY,
                 gemini_api_key=config.GEMINI_API_KEY,
                 together_api_key=config.TOGETHER_API_KEY,
                 openai_model=openai_model_to_init,
                 gemini_model=gemini_model_to_init,
                 together_model=together_model_to_init
             )
             logger.info(f"Successfully initialized client for provider: {provider}")
         except Exception as e:
             logger.error(f"Fatal Error initializing LLM client for provider '{provider}': {e}", exc_info=True)
             init_errors = True

    if init_errors:
        logger.critical("Could not initialize all required LLM clients. Exiting.")
        return


    # --- Orchestrator Initialization ---
    logger.info("Initializing Ensemble Orchestrator...")
    try:
        orchestrator = EnsembleOrchestrator(
            expert_configs=config.ENSEMBLE_EXPERTS,
            all_llm_clients=llm_clients, # Pass the dictionary of initialized clients
            wma_strategy=args.strategy,
            wma_epsilon=args.epsilon,
            wma_auto_epsilon=args.auto_epsilon,
            dataset_type=args.dataset_type,
            data_type=args.data_type,
            max_internal_refinements=args.max_refinements,
            evaluation_timeout=args.eval_timeout,
            sql_dialect=args.sql_dialect # Pass dialect
        )
    except Exception as e:
         logger.critical(f"Failed to initialize Ensemble Orchestrator: {e}", exc_info=True)
         return


    # --- Data Loading ---
    logger.info(f"Loading question data from {args.question_file}...")
    question_data = load_question_data(args.question_file)
    if not question_data: logger.critical("No question data loaded. Exiting."); return
    logger.info(f"Loaded {len(question_data)} question items.")
    logger.info(f"Loading schemas from {args.schema_file}...")
    schemas = load_schemas(args.schema_file)
    if not schemas: logger.warning("No schemas loaded. Schema-dependent steps might fail.")


    # --- Processing Loop ---
    results = []
    start_index = max(0, args.start_index)
    end_index = len(question_data)
    num_to_process = len(question_data) # Default
    if args.max_items is not None:
        if args.max_items <= 0:
             logger.warning("max_items set to 0 or negative. No items will be processed.")
             return
        end_index = min(start_index + args.max_items, len(question_data))
    num_to_process = end_index - start_index # Correct number to process

    if start_index >= len(question_data):
        logger.warning(f"Start index ({start_index}) is beyond dataset size ({len(question_data)}). Nothing to process.")
        return
    if num_to_process <= 0:
         logger.warning(f"Calculated number of items to process is {num_to_process}. Nothing to process.")
         return

    logger.info(f"Starting ensemble processing from index {start_index} to {end_index - 1} ({num_to_process} items)...")

    items_to_process = question_data[start_index:end_index]
    processed_count = 0
    # Use tqdm for progress bar
    pbar = tqdm(items_to_process, total=num_to_process, desc="Ensemble Processing", unit="item")
    for data_item in pbar:
        try:
            # Pass total rounds *intended* to be processed if using auto-epsilon
            # Alternatively, pass None and let WMA use item_counter
            total_rounds_for_auto_eps = num_to_process if args.auto_epsilon else None
            result = orchestrator.process_item(data_item, schemas, total_rounds_for_auto_eps)
            results.append(result)
            processed_count += 1
            # Update progress bar description with last status
            pbar.set_postfix({"Last Status": result.get("ensemble_status", "N/A")})
        except KeyboardInterrupt:
             logger.warning("Keyboard interrupt detected. Stopping processing...")
             break # Stop processing loop
        except Exception as e:
            # Catch unexpected errors in the main loop itself
            q_id = data_item.get('question_id', start_index + processed_count) # Get q_id for logging
            logger.critical(f"---!!! FATAL MAIN LOOP ERROR processing item around index {start_index + processed_count} (Q_ID: {q_id}): {e} !!!---", exc_info=True)
            error_result = {
                "question_id": q_id,
                "db_id": data_item.get("db_id"),
                "question": data_item.get("question"),
                "predicted_sql": f"FATAL_ERROR: {e}",
                "ensemble_status": "Fatal Main Loop Error"
                }
            results.append(error_result)
            processed_count += 1 # Count as processed to avoid infinite loop

        # Optional: Save incrementally
        if processed_count > 0 and processed_count % 10 == 0: # Adjust frequency as needed
             logger.info(f"--- Saving intermediate results ({len(results)} total items generated) ---")
             save_results(results, args.output_file)

    pbar.close() # Close tqdm bar

    # --- Saving Final Results ---
    logger.info(f"Processing finished. Generated results for {processed_count} items.")
    logger.info("Saving final results...")
    save_results(results, args.output_file)
    logger.info(f"Final results saved to {args.output_file}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
    
    
    
"""
    
# Run with default WMA strategy, processing first 10 items
python main_ensemble.py --max_items 10 --data_type bird --dataset_type dev

# Run with Randomized WMA (RWMA) and auto-epsilon
python main_ensemble.py --max_items 50 --strategy rwma --auto_epsilon

# Run Naive strategy (simple majority vote, no weight updates)
python main_ensemble.py --max_items 20 --strategy naive

# Specify output file and range
python mcp_sql_generator/main_ensemble.py --start_index 0 --max_items 1 \
    --output_file bird/output/ensemble_predictions.jsonl \
    --strategy wma --epsilon 0.005
 """