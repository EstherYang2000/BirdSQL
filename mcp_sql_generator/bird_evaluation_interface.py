# mcp_sql_generator/bird_evaluation_interface.py

import logging
import os
import sqlite3 # Keep for potential future use if needed, but rely on evaluation_utils
import traceback
from func_timeout import func_timeout, FunctionTimedOut

# --- Import functions from evaluation_utils ---
# We assume evaluation_utils.py is in a place Python can find it (e.g., same directory or in PYTHONPATH)
try:
    # This is the core function we need to call
    from evaluation_utils import execute_sql as execute_sql_comparison
    evaluation_utils_available = True
except ImportError:
    logging.error("Failed to import 'execute_sql' from 'evaluation_utils'. BIRD Evaluation will fail.", exc_info=True)
    evaluation_utils_available = False
    # Define a dummy function if import fails
    def execute_sql_comparison(*args, **kwargs):
        raise NotImplementedError("evaluation_utils.execute_sql failed to import")

logger = logging.getLogger(__name__)

# --- Comparison function (as defined in the BIRD script) ---
def calculate_ex(predicted_res, ground_truth_res):
    """
    Compares execution results using set equality.
    """
    # Basic check for non-list returns (e.g., error messages or None)
    if not isinstance(predicted_res, list) or not isinstance(ground_truth_res, list):
        return False # Cannot compare if results aren't lists

    # Normalize results by converting rows to tuples (making them hashable for set operations)
    # and sorting the outer list to handle potential order differences in set creation.
    # Note: If order *within* rows matters and isn't guaranteed by SQL, this might be too lenient.
    try:
        norm_predict = sorted([tuple(row) for row in predicted_res])
        norm_gt = sorted([tuple(row) for row in ground_truth_res])
    except TypeError:
        # Handle cases where row elements might not be directly comparable or hashable
        # Fallback to comparing string representations (less reliable)
        logger.warning("Could not convert results to tuples for set comparison, falling back to string comparison.")
        norm_predict = sorted([str(row) for row in predicted_res])
        norm_gt = sorted([str(row) for row in ground_truth_res])
    except Exception as e:
        logger.error(f"Error normalizing results for comparison: {e}", exc_info=True)
        return False # Cannot compare if normalization fails

    # Perform set comparison
    match = norm_predict == norm_gt # Compare sorted lists of tuples
    if not match:
        logger.debug(f"calculate_ex: Mismatch. Pred (norm sample): {str(norm_predict[:2])[:100]}, GT (norm sample): {str(norm_gt[:2])[:100]}")
    return match

# --- Main Evaluation Wrapper Function ---
def run_bird_evaluation(gold_sql: str, predicted_sql: str, db_path: str, timeout: float = 30.0, sql_dialect: str = "SQLite") -> bool:
    """
    Wrapper function to run the BIRD SQL evaluation with a timeout.
    Calls the execute_sql function from evaluation_utils.

    Args:
        gold_sql: The ground truth SQL query string.
        predicted_sql: The predicted SQL query string.
        db_path: Path to the database file (SQLite) or identifier for other dialects.
        timeout: Timeout in seconds for the execution.
        sql_dialect: The SQL dialect (e.g., "SQLite", "MySQL").

    Returns:
        True if the predicted SQL executes successfully, matches the gold standard results,
        and finishes within the timeout. False otherwise.
    """
    logger.debug(f"Starting BIRD evaluation wrapper. DB Path/ID: {db_path}, Dialect: {sql_dialect}, Timeout: {timeout}s")

    # Basic validation
    if not evaluation_utils_available:
        logger.error("Cannot run BIRD evaluation: evaluation_utils not imported.")
        return False
    if not gold_sql or not isinstance(gold_sql, str) or not gold_sql.strip():
        logger.warning("BIRD Evaluation skipped: Missing or invalid gold SQL.")
        return False
    if not predicted_sql or not isinstance(predicted_sql, str) or not predicted_sql.strip():
        logger.warning("BIRD Evaluation skipped: Missing or invalid predicted SQL.")
        return False
    # For SQLite, check path existence. For others, db_path might be a db name/identifier.
    if sql_dialect == "SQLite" and (not db_path or not os.path.exists(db_path)):
         logger.error(f"Database path invalid for BIRD evaluation (SQLite): {db_path}")
         return False
    elif not db_path:
         logger.error(f"Database identifier/path is missing for BIRD evaluation ({sql_dialect}).")
         return False

    try:
        # Apply timeout to the execute_sql_comparison function
        # evaluation_utils.execute_sql expects:
        # predicted_sql, ground_truth, db_path, sql_dialect, calculate_func
        match_result_code = func_timeout(
            timeout,
            execute_sql_comparison,
            args=(predicted_sql, gold_sql, db_path, sql_dialect, calculate_ex),
        )
        # Assuming execute_sql_comparison returns 1 for match (from calculate_ex), 0 otherwise
        is_match = bool(match_result_code == 1)
        logger.debug(f"BIRD evaluation result (within timeout): {is_match} (Result code: {match_result_code})")
        return is_match

    except FunctionTimedOut:
        logger.warning(f"BIRD evaluation timed out ({timeout}s). DB: {db_path}, SQL: {predicted_sql[:200]}...")
        return False
    except KeyboardInterrupt:
         logger.warning("Keyboard interrupt during BIRD evaluation.")
         raise # Re-raise interrupt
    except Exception as e:
        # Catch errors from within execute_sql_comparison (DB connection, SQL execution, comparison)
        logger.error(f"Error during BIRD evaluation execution/comparison: {e}", exc_info=True)
        logger.debug(f"Failed SQL (Pred): {predicted_sql[:200]}...")
        logger.debug(f"Failed SQL (Gold): {gold_sql[:200]}...")
        return False