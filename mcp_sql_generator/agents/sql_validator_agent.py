# mcp_sql_generator/agents/sql_validator_agent.py
from utils import execute_sql, get_db_path

class SQLValidatorAgent:
    def __init__(self, dataset_type: str = "dev", data_type: str = "bird"):
        # Store dataset/data type if needed for get_db_path variations
        self.dataset_type = dataset_type
        self.data_type = data_type
        print(f"Initialized SQLValidatorAgent (Dataset Type: {dataset_type}, Data Type: {data_type})")

    def validate_sql(self, sql: str, db_id: str) -> tuple[bool, str | list | None]:
        """
        Validates SQL syntax by attempting execution.
        Returns: (success, result_or_error_message)
        """
        print(f"  [Validator Agent] Validating SQL for DB: {db_id}...")
        db_path = get_db_path(db_id, self.dataset_type, self.data_type)
        if not db_path:
             error_msg = f"Could not find database path for db_id: {db_id}"
             print(f"  [Validator Agent] {error_msg}")
             return False, error_msg

        success, result = execute_sql(sql, db_path)
        if success:
            print(f"  [Validator Agent] SQL Syntax OK. Execution successful.")
            # We might get results (list) or None for non-select queries
            # For critique purposes, just knowing it ran is often enough,
            # but we return the result/error anyway.
        else:
            print(f"  [Validator Agent] SQL Syntax/Execution Error: {result}")

        return success, result