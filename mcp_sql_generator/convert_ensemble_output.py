# convert_ensemble_output.py (Revised to output JSON dictionary)
import json
import argparse
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_sql(sql_string):
    """Removes leading/trailing whitespace and replaces internal newlines/tabs."""
    if not isinstance(sql_string, str):
        return ""
    cleaned = ' '.join(sql_string.split()).strip()
    return cleaned

def convert_to_json_output(input_jsonl_path, pred_output_json_path):
    """
    Reads the ensemble output JSONL file and creates a JSON dictionary file
    in the format expected by the BIRD evaluation script's package_sqls (mode='pred').
    Uses question_id as the key.
    """
    if not os.path.exists(input_jsonl_path):
        logger.error(f"Input file not found: {input_jsonl_path}")
        return

    # Ensure output directory exists
    pred_output_dir = os.path.dirname(pred_output_json_path)
    if pred_output_dir and not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir, exist_ok=True)
        logger.info(f"Created directory: {pred_output_dir}")

    output_data = {}
    processed_lines = 0
    skipped_lines = 0
    missing_q_id_lines = 0

    try:
        with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                try:
                    data = json.loads(line)

                    pred_sql = data.get("predicted_sql")
                    db_id = data.get("db_id")
                    # --- Get question_id, needed as key ---
                    q_id = data.get("question_id")

                    # --- Input Validation ---
                    if q_id is None: # Check for existence and None explicitly
                         logger.warning(f"Skipping line {i+1}: Missing 'question_id'.")
                         missing_q_id_lines += 1
                         skipped_lines += 1
                         continue
                    if not db_id or not isinstance(db_id, str) or not db_id.strip():
                        logger.warning(f"Skipping line {i+1} (Q_ID: {q_id}): Missing or invalid 'db_id'.")
                        skipped_lines += 1
                        continue

                    # --- Handle Predicted SQL ---
                    # Use a placeholder if prediction failed, but still include the db_id
                    if not pred_sql or not isinstance(pred_sql, str) or pred_sql.strip().startswith("Error:"):
                        logger.warning(f"Line {i+1} (Q_ID: {q_id}): Using placeholder 'SELECT 1' for missing/error predicted SQL. Original: '{pred_sql}'")
                        pred_sql_cleaned = "SELECT 1" # Use a valid, simple SQL as placeholder
                    else:
                        pred_sql_cleaned = clean_sql(pred_sql)

                    # --- Format the value string as expected by package_sqls ---
                    # Combine cleaned SQL and db_id with the special delimiter
                    output_value_string = f"{pred_sql_cleaned}\t----- bird -----\t{db_id}"

                    # Use question_id as the key (convert to string if it's numeric)
                    output_key = str(q_id)
                    if output_key in output_data:
                         logger.warning(f"Duplicate question_id '{output_key}' found on line {i+1}. Overwriting previous entry.")
                    output_data[output_key] = output_value_string
                    processed_lines += 1

                except json.JSONDecodeError:
                    logger.error(f"Skipping invalid JSON line {i+1}: {line.strip()}")
                    skipped_lines += 1
                except Exception as e:
                     logger.error(f"Error processing line {i+1}: {e}", exc_info=True)
                     skipped_lines += 1

        # --- Write the collected data as a single JSON object ---
        with open(pred_output_json_path, 'w', encoding='utf-8') as outfile:
             json.dump(output_data, outfile, indent=4) # Use indent for readability

        logger.info(f"Conversion complete.")
        logger.info(f"Processed {processed_lines} lines.")
        if skipped_lines > 0:
             logger.warning(f"Skipped {skipped_lines} lines due to errors or missing data (including {missing_q_id_lines} missing Q_IDs).")
        logger.info(f"Predicted SQL JSON dictionary written to: {pred_output_json_path}")

    except IOError as e:
        logger.error(f"File I/O error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ensemble output JSONL to BIRD prediction JSON dictionary.")
    parser.add_argument('--input_file', type=str, required=True,
                        help="Path to the input ensemble predictions JSONL file (e.g., ensemble_predictions.jsonl).")
    # --- Modified output argument name and help text ---
    parser.add_argument('--pred_output_json', type=str, required=True,
                        help="Path to write the predicted SQL JSON dictionary file (e.g., bird_predictions.json).")

    args = parser.parse_args()

    # --- Modified function call ---
    convert_to_json_output(args.input_file, args.pred_output_json)
    
    """
    
    python mcp_sql_generator/convert_ensemble_output.py \
        --input_file bird/output/ensemble_predictions.jsonl \
        --pred_output_json bird/output/predictions_for_eval.json    
    """