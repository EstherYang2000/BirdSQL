# mcp_sql_generator/utils.py

import json
import re
import os
import sqlite3
import logging
import config  # Import config to access default paths if needed

# Set up logger for this module
logger = logging.getLogger(__name__)

# --- Data Loading Functions ---

def load_question_data(filepath: str) -> list[dict]:
    """
    Loads question data from a JSON file.
    Expects the file to contain a JSON list of objects,
    or optionally an object with a 'questions' key holding the list.
    """
    data = []
    logger.debug(f"Attempting to load question data from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict) and "questions" in content and isinstance(content["questions"], list):
                logger.warning("Loaded data from 'questions' key in the JSON object.")
                data = content["questions"]
            else:
                logger.error(f"Expected a JSON list or object with 'questions' key in {filepath}, but found {type(content)}")
    except FileNotFoundError:
        logger.error(f"Question data file not found at {filepath}")
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred loading question data from {filepath}: {e}", exc_info=True)

    if data:
        logger.info(f"Successfully loaded {len(data)} question items from {filepath}")
    else:
        logger.warning(f"No question data loaded from {filepath}")
    return data

def load_schemas(filepath: str) -> dict[str, dict]:
    """
    Loads schema data from a JSON file (e.g., tables.json) and organizes it
    into a dictionary keyed by db_id.
    """
    schemas = {}
    logger.debug(f"Attempting to load schema data from: {filepath}")
    # Reuse logic similar to load_question_data as tables.json is often a list
    raw_schemas = load_question_data(filepath) # Use the same robust loading

    if not raw_schemas:
        logger.warning(f"No schema data loaded from {filepath}")
        return {}

    for schema_info in raw_schemas:
        if not isinstance(schema_info, dict):
            logger.warning(f"Skipping non-dictionary item found in schema file: {type(schema_info)}")
            continue
        db_id = schema_info.get("db_id")
        if db_id:
            if db_id in schemas:
                 logger.warning(f"Duplicate db_id '{db_id}' found in schema file. Overwriting previous entry.")
            schemas[db_id] = schema_info
        else:
            logger.warning("Found schema entry without a 'db_id', skipping.")

    logger.info(f"Successfully loaded schemas for {len(schemas)} databases from {filepath}")
    return schemas

# --- Schema Formatting Function ---

def format_schema(schema_data: dict, db_id: str) -> str:
    """
    Formats the structured schema information from a schema dictionary
    into a readable string format suitable for LLM prompts.

    Args:
        schema_data: The dictionary containing schema info for one database.
        db_id: The database identifier.

    Returns:
        A formatted string representation of the schema.
    """
    if not schema_data or not isinstance(schema_data, dict):
        logger.warning(f"Invalid or empty schema_data provided for db_id '{db_id}'")
        return f"-- Schema information unavailable or invalid for database: {db_id} --"

    formatted_string = f"### Database Schema: {db_id}\n"

    # Safely get schema components with defaults
    table_names = schema_data.get("table_names_original", [])
    column_data = schema_data.get("column_names_original", []) # List of [table_idx, col_name]
    column_types = schema_data.get("column_types", [])
    pk_indices = schema_data.get("primary_keys", []) # Indices referring to column_data
    fk_pairs = schema_data.get("foreign_keys", []) # Pairs of indices [col_idx_from, col_idx_to]

    if not table_names or not column_data:
         logger.warning(f"Schema for '{db_id}' is missing table names or column data.")
         return formatted_string + "-- Missing table or column information --"

    # --- Map column index to details for easier lookup ---
    column_details = {} # Map index -> {name, table_name, type}
    for i, (table_idx, col_name) in enumerate(column_data):
        # Skip wildcard column often present at index 0
        if table_idx == -1 and col_name == '*':
            continue
        if 0 <= table_idx < len(table_names):
            table_name = table_names[table_idx]
            col_type = column_types[i].upper() if i < len(column_types) else 'UNKNOWN_TYPE'
            column_details[i] = {"name": col_name, "table": table_name, "type": col_type}
        else:
            logger.warning(f"Invalid table_idx {table_idx} for column '{col_name}' (index {i}) in db '{db_id}'.")

    # --- Identify Primary Keys ---
    primary_key_cols = set() # Store indices of PK columns
    for pk_index in pk_indices:
         # Handle composite keys which might be lists/tuples
        if isinstance(pk_index, (list, tuple)):
            for sub_index in pk_index:
                 if sub_index in column_details:
                     primary_key_cols.add(sub_index)
                 else:
                      logger.warning(f"PK sub-index {sub_index} not found in column details for db '{db_id}'.")
        elif isinstance(pk_index, int):
             if pk_index in column_details:
                 primary_key_cols.add(pk_index)
             else:
                  logger.warning(f"PK index {pk_index} not found in column details for db '{db_id}'.")
        else:
             logger.warning(f"Unexpected PK format: {pk_index} in db '{db_id}'.")


    # --- Format Tables and Columns ---
    formatted_string += "Tables:\n"
    for table_idx, table_name in enumerate(table_names):
        formatted_string += f"- {table_name} (\n"
        table_cols_str = []
        for col_idx, details in column_details.items():
            if details["table"] == table_name:
                pk_marker = " PK" if col_idx in primary_key_cols else ""
                table_cols_str.append(f"    {details['name']} ({details['type']}){pk_marker}")
        if table_cols_str:
            formatted_string += ",\n".join(table_cols_str)
        else:
             formatted_string += "    -- No columns found or mapped --"
        formatted_string += "\n  )\n"


    # --- Format Foreign Keys ---
    if fk_pairs:
        formatted_string += "\nForeign Keys:\n"
        for fk_col_idx, pk_col_idx in fk_pairs:
            if fk_col_idx in column_details and pk_col_idx in column_details:
                fk_detail = column_details[fk_col_idx]
                pk_detail = column_details[pk_col_idx]
                formatted_string += (f"- {fk_detail['table']}({fk_detail['name']}) REFERENCES "
                                     f"{pk_detail['table']}({pk_detail['name']})\n")
            else:
                 logger.warning(f"Could not resolve FK relationship involving indices {fk_col_idx} -> {pk_col_idx} in db '{db_id}'.")
                 formatted_string += f"- -- Error resolving FK: {fk_col_idx} -> {pk_col_idx} --\n"
    else:
         formatted_string += "\nForeign Keys: None\n"


    return formatted_string.strip()

# --- Result Saving Function ---

def save_results(results: list[dict], output_filepath: str):
    """Saves the results list to a JSON Lines file."""
    logger.info(f"Attempting to save {len(results)} results to: {output_filepath}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as f:
            for item in results:
                try:
                    # Ensure complex objects (like exceptions sometimes captured) are serializable
                    json_string = json.dumps(item, ensure_ascii=False, default=str) # Use default=str as fallback
                    f.write(json_string + '\n')
                except TypeError as e:
                    logger.error(f"Could not serialize item to JSON: {e}. Item: {str(item)[:500]}...") # Log partial item
                    # Optionally write a placeholder error to the file
                    error_item = {"error": "Serialization failed", "original_item_preview": str(item)[:500]}
                    f.write(json.dumps(error_item, ensure_ascii=False) + '\n')

        logger.info(f"Results successfully saved to {output_filepath}")
    except IOError as e:
        logger.error(f"Error writing results to {output_filepath}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving results: {e}", exc_info=True)


# --- SQL Parsing Function ---

def parse_sql_from_response(response_text: str) -> str:
    """
    Extracts the SQL query from the LLM response, handling potential markdown blocks
    and preliminary explanatory text. Attempts to return only the SQL statement.
    """
    if not isinstance(response_text, str):
        return "" # Return empty if input is not a string

    text = response_text.strip()

    # 1. Look for ```sql ... ``` blocks
    sql_match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
        # Remove trailing semicolon often added by LLMs
        return sql.rstrip(';')

    # 2. Look for ``` ... ``` blocks
    sql_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if sql_match:
        sql_content = sql_match.group(1).strip()
        # Basic check if it looks like SQL
        if sql_content.upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER')):
             return sql_content.rstrip(';')
        else:
             # If content doesn't look like SQL, continue searching
             pass

    # 3. Look for lines starting with SQL keywords, assuming minimal preamble
    lines = text.splitlines()
    sql_lines = []
    found_sql_start = False
    sql_keywords = ('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER')
    for line in lines:
        stripped_line = line.strip()
        if not found_sql_start and stripped_line.upper().startswith(sql_keywords):
            found_sql_start = True
        if found_sql_start:
            # Stop if we encounter what looks like explanation again
            if not stripped_line: # Skip empty lines between SQL parts
                 continue
            # Heuristic: if a line doesn't start with common SQL continuation patterns and isn't just punctuation
            # maybe it's explanation - this is tricky and imperfect.
            # if len(sql_lines) > 0 and not stripped_line.upper().startswith(('FROM','WHERE','GROUP','ORDER','LIMIT','JOIN','ON','UNION','INTERSECT','EXCEPT', ')', ';', ',')):
            #     break # Removed this heuristic, might be too aggressive

            sql_lines.append(stripped_line)

    if sql_lines:
        potential_sql = "\n".join(sql_lines).strip()
        return potential_sql.rstrip(';')

    # 4. Fallback: Check if the entire cleaned response starts with SQL keywords
    # Remove potential leading/trailing quotes if LLM wrapped the whole SQL
    if text.startswith(('"', "'")) and text.endswith(('"', "'")) and text[0] == text[-1]:
        text = text[1:-1].strip()
    if text.upper().startswith(sql_keywords):
         return text.rstrip(';')

    # 5. If no SQL found after all attempts
    logger.warning(f"Could not reliably parse SQL from response:\n---\n{response_text}\n---")
    # Return the original cleaned text as it might be an error message or unexpected format
    return response_text.strip()


# --- Database Path and Execution Functions ---

def get_db_path(db_id: str, dataset_type: str = "dev", data_type: str = "bird") -> str | None:
    """
    Constructs the path to the SQLite database file based on configuration.

    Args:
        db_id: The database identifier.
        dataset_type: 'dev' or 'test'.
        data_type: 'bird' or 'spider'.

    Returns:
        The absolute or relative path string to the database file, or None if not found.
    """
    base_dir = ""
    db_sub_dir = ""

    # Determine base directory and sub-directory based on data type and set
    try:
        if data_type.lower() == "spider":
            base_dir = config.DEFAULT_SPIDER_DATA_DIR if hasattr(config, 'DEFAULT_SPIDER_DATA_DIR') else "data/spider/"
            db_sub_dir = "database" if dataset_type.lower() == "dev" else "test_database"
        elif data_type.lower() == "bird":
            base_dir = config.DEFAULT_BIRD_DATA_DIR if hasattr(config, 'DEFAULT_BIRD_DATA_DIR') else "bird/bird/"
            # Bird often uses 'database' for both dev/test based on examples seen
            db_sub_dir = "database"
        else:
            logger.error(f"Unsupported data_type '{data_type}' for DB path construction.")
            return None
    except AttributeError as e:
         logger.error(f"Configuration error accessing default data directories: {e}. Please define them in config.py")
         return None

    # Construct the expected path
    constructed_path = os.path.join(base_dir, db_sub_dir, db_id, f"{db_id}.sqlite")
    logger.debug(f"Constructed DB path: {constructed_path}")

    # Check if the constructed path exists
    if os.path.exists(constructed_path):
        return constructed_path
    else:
        logger.warning(f"Database path not found at expected location: {constructed_path}")
        # Add fallbacks if necessary, e.g., check alternative config paths
        # alt_path = os.path.join(config.ALTERNATIVE_DB_DIR, ...)
        # if os.path.exists(alt_path): return alt_path
        return None


def execute_sql(sql: str, db_path: str) -> tuple[bool, str | list | None]:
    """
    Execute a SQL query against the target SQLite database file.

    Args:
        sql: The SQL query string to execute.
        db_path: The file path to the SQLite database.

    Returns:
        tuple[bool, str | list | None]:
            - bool: True if execution was successful (syntax OK), False otherwise.
            - str | list | None:
                - If success and SELECT: A list of fetched rows (limited number).
                - If success and not SELECT: None.
                - If failure: An error message string.
    """
    if not db_path or not os.path.exists(db_path):
        error_msg = f"Database path does not exist or is invalid: {db_path}"
        logger.error(error_msg)
        return False, error_msg
    if not sql or not isinstance(sql, str) or not sql.strip():
        error_msg = "Empty or invalid SQL query provided for execution."
        logger.error(error_msg)
        return False, error_msg

    conn = None
    try:
        # Connect to the database file
        # Consider read-only mode for SELECTs if safety is paramount:
        # conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logger.debug(f"Executing SQL on {db_path}: {sql[:200]}...")
        cursor.execute(sql)

        # Determine if it was a SELECT statement (heuristic)
        is_select = sql.strip().upper().startswith("SELECT")

        if is_select:
            # Fetch a limited number of results for SELECT queries
            rows = cursor.fetchmany(config.DB_EXEC_FETCH_LIMIT if hasattr(config, 'DB_EXEC_FETCH_LIMIT') else 5) # Configurable limit
            logger.debug(f"Execution successful (SELECT). Fetched {len(rows)} rows.")
            return True, rows
        else:
            # For non-SELECT (INSERT, UPDATE, DELETE, etc.), commit changes
            conn.commit()
            logger.debug("Execution successful (non-SELECT). Changes committed.")
            return True, None # Success, no data rows returned

    except sqlite3.Error as e:
        # Specific SQLite errors
        error_msg = f"SQLite Error: {str(e).replace(chr(10), ' ')}" # Replace newline for cleaner logs
        logger.warning(f"SQL Execution Failed. DB: {os.path.basename(db_path)}, Error: {error_msg}, SQL: {sql[:200]}...")
        return False, error_msg
    except Exception as e:
        # Catch other potential errors during execution
        error_msg = f"Unexpected Execution Error: {str(e)}"
        logger.error(f"SQL Execution Failed. DB: {os.path.basename(db_path)}, Error: {error_msg}, SQL: {sql[:200]}...", exc_info=True)
        return False, error_msg
    finally:
        # Ensure the connection is closed
        if conn:
            conn.close()
            logger.debug(f"Database connection closed for {db_path}")