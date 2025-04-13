# mcp_sql_generator/ensemble_orchestrator.py
import time
import config
import logging
import os

# LLM and Agent Imports
from llm_interface import LLMInterface, get_llm_client
from agents.planner_agent import PlannerAgent
from agents.executor_agent import ExecutorAgent
from agents.sql_validator_agent import SQLValidatorAgent # Used in internal MCP
from agents.critique_agent import CritiqueAgent
from agents.refiner_agent import RefinerAgent

# WMA Import
from wma import WeightedMajorityAlgorithm, auto_select_epsilon

# --- Import the NEW BIRD evaluation interface ---
try:
    # Assumes bird_evaluation_interface.py is in the same directory or accessible via PYTHONPATH
    from bird_evaluation_interface import run_bird_evaluation
    evaluation_available = True
    logger_eval_interface = logging.getLogger("bird_evaluation_interface") # Get logger used within the interface if needed
except ImportError as e:
    logging.error(f"Failed to import BIRD evaluation functions from bird_evaluation_interface: {e}. Evaluation disabled.", exc_info=True)
    evaluation_available = False
    # Define dummy function if import fails to prevent crashes
    def run_bird_evaluation(*args, **kwargs):
        logging.error("run_bird_evaluation called but interface failed to import.")
        return False

# Utility Imports
from utils import format_schema, parse_sql_from_response, get_db_path

logger = logging.getLogger(__name__) # Logger for this module

class EnsembleOrchestrator:
    """
    Orchestrates the generation of SQL candidates using an ensemble of experts,
    where each expert runs an internal MCP process. Uses WMA for final selection
    and BIRD-style evaluation for weight updates.
    """
    def __init__(self, expert_configs: list[dict], all_llm_clients: dict[str, LLMInterface],
                 wma_strategy: str = config.DEFAULT_WMA_STRATEGY,
                 wma_epsilon: float = config.DEFAULT_WMA_EPSILON,
                 wma_auto_epsilon: bool = config.DEFAULT_WMA_AUTO_EPSILON,
                 dataset_type: str = "dev", data_type: str = "bird",
                 max_internal_refinements: int = config.DEFAULT_MAX_REFINEMENTS,
                 evaluation_timeout: float = config.DEFAULT_EVALUATION_TIMEOUT,
                 sql_dialect: str = "SQLite"):

        self.expert_configs = expert_configs
        self.all_llm_clients = all_llm_clients
        self.dataset_type = dataset_type
        self.data_type = data_type
        self.wma_strategy = wma_strategy
        self.wma_auto_epsilon = wma_auto_epsilon
        self.initial_epsilon = wma_epsilon
        self.max_internal_refinements = max_internal_refinements
        self.evaluation_timeout = evaluation_timeout
        self.sql_dialect = sql_dialect
        self.item_counter = 0

        # Initialize WMA Algorithm
        self.wma = WeightedMajorityAlgorithm(experts_config=self.expert_configs, epsilon=self.initial_epsilon)

        # Initialize Validator Agent (used within expert MCP)
        self.validator = SQLValidatorAgent(dataset_type, data_type)

        # Check if evaluation tools (via interface) are available
        self.evaluation_enabled = evaluation_available
        if not self.evaluation_enabled:
            logger.warning("BIRD Evaluation disabled (interface import failed). WMA weight updates based on correctness will be skipped.")
        else:
            logger.info("BIRD Evaluation interface imported successfully.")

        logger.info(f"EnsembleOrchestrator initialized. Strategy: {self.wma_strategy}, Internal Refinements: {self.max_internal_refinements}, Evaluation Enabled: {self.evaluation_enabled}, Dialect: {self.sql_dialect}")


    def _get_llm_client_for_expert(self, expert_name: str) -> LLMInterface | None:
        """Finds the expert config and returns the corresponding LLM client."""
        for conf in self.expert_configs:
            if conf.get("name") == expert_name:
                provider = conf.get("provider")
                if provider and provider in self.all_llm_clients:
                    return self.all_llm_clients[provider]
                else:
                    logger.error(f"LLM Client for provider '{provider}' not found for expert '{expert_name}'.")
                    return None
        logger.error(f"Configuration for expert '{expert_name}' not found.")
        return None


    def _run_expert_mcp_process(self, expert_name: str, expert_llm_client: LLMInterface,
                               question: str, schema_string: str, evidence: str, db_id: str) -> tuple[str, list, str]:
        """
        Runs the internal Plan -> Execute -> Validate -> Critique -> Refine loop for a single expert.
        Returns the final refined SQL, critiques generated, and status string.
        """
        logger.info(f"--- Running internal MCP for Expert: {expert_name} ---")
        plan, initial_sql, current_sql, critiques, final_status, last_successful_sql = "", "", "Error: MCP start", [], "MCP Started", "Error: No success"

        # Instantiate agents using the specific expert's LLM client
        planner = PlannerAgent(expert_llm_client)
        executor = ExecutorAgent(expert_llm_client)
        critique_agent = CritiqueAgent(expert_llm_client)
        refiner = RefinerAgent(expert_llm_client)

        try:
            # 1. Plan
            plan = planner.generate_plan(question, schema_string, evidence)
            if plan.startswith("Error:"): raise ValueError(f"Planner failed: {plan}")
            logger.debug(f"Expert '{expert_name}' - Plan generated.")

            # 2. Execute
            initial_sql = executor.execute_plan(question, schema_string, evidence, plan)
            if initial_sql.startswith("Error:"): raise ValueError(f"Executor failed: {initial_sql}")
            current_sql = initial_sql
            last_successful_sql = current_sql
            final_status = "MCP Executed"
            logger.debug(f"Expert '{expert_name}' - Initial SQL: {current_sql[:100]}...")

            # 3. Refinement Loop
            for i in range(self.max_internal_refinements):
                logger.debug(f"  Expert '{expert_name}' - Internal Refinement Cycle {i+1}/{self.max_internal_refinements}")

                # 3a. Validate SQL using the shared validator
                validation_success, validation_output = self.validator.validate_sql(current_sql, db_id)
                validation_result = (validation_success, validation_output)
                logger.debug(f"  Expert '{expert_name}' - Validation result: {validation_success}, Output/Error: {str(validation_output)[:100]}...")

                # 3b. Critique
                critique = critique_agent.critique_sql(question, schema_string, evidence, plan, current_sql, validation_result)
                critiques.append(critique)
                if critique.startswith("Error:"):
                    logger.warning(f"Expert '{expert_name}' critique failed in cycle {i+1}. Using previous SQL.")
                    final_status = f"MCP Completed - Critique Error Cycle {i+1}"
                    current_sql = last_successful_sql # Revert to last good one
                    break # Stop this expert's refinement

                # 3c. Check Convergence
                critique_lower = critique.lower()
                if validation_success and ("looks correct" in critique_lower or "query is correct" in critique_lower or "no issues found" in critique_lower):
                    logger.debug(f"Expert '{expert_name}' - Validation OK & Critique positive. Stopping internal refinement.")
                    final_status = f"MCP Completed - Converged Cycle {i+1}"
                    break # Stop this expert's refinement

                # 3d. Refine (Only if needed based on validation or critique)
                if not validation_success or not ("looks correct" in critique_lower or "query is correct" in critique_lower or "no issues found" in critique_lower):
                    logger.debug(f"Expert '{expert_name}' - Triggering refinement...")
                    refined_sql_attempt = refiner.refine_sql(question, schema_string, evidence, plan, current_sql, critique)

                    if refined_sql_attempt.startswith("Error:"):
                        logger.warning(f"Expert '{expert_name}' refinement failed in cycle {i+1}. Using previous SQL.")
                        final_status = f"MCP Completed - Refinement Error Cycle {i+1}"
                        current_sql = last_successful_sql
                        break # Stop this expert's refinement
                    elif not refined_sql_attempt: # Check for empty string
                        logger.warning(f"Expert '{expert_name}' refinement empty in cycle {i+1}. Using previous SQL.")
                        final_status = f"MCP Completed - Empty Refinement Cycle {i+1}"
                        current_sql = last_successful_sql
                        break # Stop this expert's refinement
                    elif refined_sql_attempt == current_sql:
                         logger.debug(f"Expert '{expert_name}' - SQL unchanged in cycle {i+1}. Stopping internal refinement.")
                         final_status = f"MCP Completed - No Change Cycle {i+1}"
                         break # Stop this expert's refinement
                    else:
                         # Refinement successful and changed SQL
                         current_sql = refined_sql_attempt
                         last_successful_sql = current_sql # Update last known good state
                         logger.debug(f"Expert '{expert_name}' - Refined SQL: {current_sql[:100]}...")
                else:
                    # If validation passed and critique was positive, we already broke in 3c
                    logger.debug(f"Expert '{expert_name}' - No refinement needed in cycle {i+1}.")
                    # We might break here too if critique wasn't strongly negative
                    # break

            # If loop finished without break (max refinements reached)
            else:
                final_status = f"MCP Completed - Max Refinements Reached ({self.max_internal_refinements})"

        except ValueError as ve: # Catch specific errors from Plan/Execute failures
             logger.error(f"MCP Value Error for expert '{expert_name}': {ve}")
             final_status = f"MCP Failed - {ve}"
             current_sql = "Error: MCP Step Failed" # Ensure SQL indicates failure
        except Exception as e:
            logger.error(f"Unexpected Error during MCP process for expert '{expert_name}': {e}", exc_info=True)
            final_status = f"MCP Error - {type(e).__name__}"
            # Use last known good SQL if available, otherwise indicate general MCP error
            current_sql = last_successful_sql if 'last_successful_sql' in locals() and not last_successful_sql.startswith("Error:") else "Error: MCP Exception"

        logger.info(f"--- Finished internal MCP for Expert: {expert_name}. Final Status: {final_status} ---")
        return plan,current_sql, critiques, final_status # Return final SQL, critiques, status


    def process_item(self, data_item: dict, schemas: dict, total_rounds: int | None) -> dict:
        """ Orchestrates the ensemble generation (with internal MCP) and WMA voting using BIRD evaluation. """
        self.item_counter += 1
        q_id = data_item.get("question_id", f"unk_{self.item_counter}")
        db_id = data_item.get("db_id")
        question = data_item.get("question")
        evidence = data_item.get("evidence", "")
        gold_sql = data_item.get("gold_sql") or data_item.get("SQL") or data_item.get("query")

        logger.info(f"=== Processing Item {self.item_counter} (ID: {q_id}, DB: {db_id}) ===")
        start_time = time.time()

        # --- Schema Prep ---
        if not db_id or not question:
             logger.error(f"Skipping item {q_id}: Missing 'db_id' or 'question'."); return {**data_item, "predicted_sql": "Error: Missing db_id or question", "ensemble_status": "Skipped - Missing Input"}
        schema_data = schemas.get(db_id)
        if not schema_data:
             logger.error(f"Skipping item {q_id}: Schema for '{db_id}' not found."); return {**data_item, "predicted_sql": f"Error: Schema not found for {db_id}", "ensemble_status": "Skipped - No Schema"}
        schema_string = format_schema(schema_data, db_id)

        # --- Get DB Path (needed for evaluation) ---
        # If dialect is SQLite, path is crucial. For others, db_id might suffice for connection.
        db_path = get_db_path(db_id, self.dataset_type, self.data_type)
        eval_db_identifier = db_path if self.sql_dialect == "SQLite" else db_id # Use path for SQLite, id for others
        if self.sql_dialect == "SQLite" and not db_path:
             logger.error(f"Skipping item {q_id}: SQLite Database path for '{db_id}' could not be resolved.")
             # Disable evaluation for this item if path is missing but dialect is SQLite
             can_evaluate_this_item = False
             # Allow processing to continue but evaluation will be skipped
             # return {**data_item, "predicted_sql": f"Error: DB path not found for {db_id}", "ensemble_status": "Skipped - No DB Path"}
        else:
             can_evaluate_this_item = True # Assume evaluation possible if path found or not needed


        # --- Candidate Generation (Each expert runs internal MCP) ---
        predictions_dict: dict[str, list[str]] = {}
        expert_mcp_details = {}
        for expert_conf in self.expert_configs:
            expert_name = expert_conf["name"]
            expert_llm_client = self._get_llm_client_for_expert(expert_name)
            if not expert_llm_client:
                logger.error(f"Skipping expert '{expert_name}'."); predictions_dict[expert_name] = ["Error: LLM Client unavailable"]; expert_mcp_details[expert_name] = {'status': 'Skipped - No Client', 'critiques': []}; continue

            plan,refined_sql_candidate, internal_critiques, internal_status = self._run_expert_mcp_process(
                expert_name, expert_llm_client, question, schema_string, evidence, db_id
            )
            predictions_dict[expert_name] = [refined_sql_candidate] # Store single refined result
            expert_mcp_details[expert_name] = {'initial_plan':plan,'status': internal_status, 'critiques': internal_critiques}
            if refined_sql_candidate.startswith("Error:"): logger.warning(f"Expert '{expert_name}' MCP error: {refined_sql_candidate}")


        # --- WMA Voting ---
        final_sql = None; chosen_experts = []; vote_score = 0.0
        valid_predictions_for_vote = { name: preds for name, preds in predictions_dict.items() if preds and not preds[0].startswith("Error:") }
        if not valid_predictions_for_vote:
             final_sql = "Error: All experts failed internal MCP"
             logger.error(f"No valid refined SQL candidates for WMA voting for item {q_id}.")
        else:
            if self.wma_strategy == "wma": final_sql, chosen_experts, vote_score = self.wma.weighted_majority_vote(valid_predictions_for_vote)
            elif self.wma_strategy == "rwma": final_sql, chosen_experts, vote_score = self.wma.randomized_weighted_majority_vote(valid_predictions_for_vote)
            elif self.wma_strategy == "naive": final_sql, chosen_experts, vote_score = self.wma.naive_vote(valid_predictions_for_vote)
            if final_sql is None: final_sql = "Error: WMA Voting Failed"; logger.error(f"WMA ({self.wma_strategy}) voting failed despite valid candidates for item {q_id}.")


        # --- Evaluation for Weight Update using BIRD evaluation ---
        expert_contributions = {conf["name"]: False for conf in self.expert_configs}
        is_final_correct = None

        # Perform evaluation ONLY if enabled AND gold exists AND db identifier is valid for the dialect
        if self.evaluation_enabled and gold_sql and eval_db_identifier and can_evaluate_this_item:
            logger.debug(f"Performing BIRD evaluation on refined candidates for item {q_id}...")
            any_correct_overall = False

            for expert_name, refined_candidate_list in predictions_dict.items():
                 if refined_candidate_list and not refined_candidate_list[0].startswith("Error:"):
                     refined_sql = refined_candidate_list[0]
                     try:
                         is_expert_correct = run_bird_evaluation(
                             gold_sql=gold_sql,
                             predicted_sql=refined_sql,
                             db_path=eval_db_identifier, # Use path for SQLite, db_id for others
                             timeout=self.evaluation_timeout,
                             sql_dialect=self.sql_dialect # Pass the dialect
                         )
                         if is_expert_correct:
                             expert_contributions[expert_name] = True
                             any_correct_overall = True
                             logger.info(f"Correct refined candidate FOUND from expert '{expert_name}'.")
                     except Exception as e:
                          logger.error(f"Error calling run_bird_evaluation for expert '{expert_name}', db '{db_id}': {e}", exc_info=True)

            # Update WMA weights based on contributions
            self.wma.update_weights(expert_contributions, self.wma_strategy)

            # Evaluate the FINAL chosen SQL for reporting
            if final_sql and not final_sql.startswith("Error:"):
                try:
                    is_final_correct = run_bird_evaluation(
                        gold_sql=gold_sql,
                        predicted_sql=final_sql,
                        db_path=eval_db_identifier,
                        timeout=self.evaluation_timeout,
                        sql_dialect=self.sql_dialect
                    )
                    logger.info(f"Item {q_id}: Final WMA-selected SQL Correctness = {is_final_correct}")
                except Exception as e:
                     logger.error(f"Error evaluating final selected SQL using BIRD evaluation: {e}", exc_info=True)
                     is_final_correct = False
            else:
                 is_final_correct = False # Mark as incorrect if no final SQL or it's an error string

        else:
            # Log skip reason
            is_final_correct = None
            log_skip_reason = "Evaluation skipped: "
            if not self.evaluation_enabled: log_skip_reason += "Import failed. "
            if not gold_sql: log_skip_reason += "Missing gold SQL. "
            if not eval_db_identifier: log_skip_reason += "Missing DB identifier/path. "
            if not can_evaluate_this_item: log_skip_reason += f"Cannot evaluate for {db_id} (e.g., SQLite path missing). "
            logger.warning(log_skip_reason.strip())
            # Naive mistake update if needed (based purely on generation success)
            if self.wma_strategy == 'naive':
                 naive_contributions = {name: (preds and not preds[0].startswith("Error:")) for name, preds in predictions_dict.items()}
                 self.wma.update_weights(naive_contributions, self.wma_strategy)

        # --- Auto Epsilon Update ---
        current_epsilon = self.wma.epsilon
        if self.wma_auto_epsilon and self.item_counter > 0:
            num_experts = len(self.expert_configs); mistake_counts = self.wma.get_mistake_counts()
            best_mistake_count = min(mistake_counts.values()) if mistake_counts else 0
            new_epsilon = auto_select_epsilon(num_experts, num_rounds=self.item_counter, best_mistakes=best_mistake_count)
            self.wma.set_epsilon(new_epsilon); current_epsilon = new_epsilon

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        # --- Result Packaging ---
        result = {
            "question_id": q_id, "db_id": db_id, "question": question, "gold_sql": gold_sql,
            "predicted_sql": final_sql if final_sql else "Error: No SQL selected",
            "ensemble_strategy": self.wma_strategy, "ensemble_chosen_experts": chosen_experts,
            "ensemble_vote_score": vote_score, "ensemble_final_sql_correct": is_final_correct,
            "expert_refined_candidates": predictions_dict, # Contains the single refined SQL (or error) from each expert
            "expert_mcp_details": expert_mcp_details, # Contains internal status/critiques
            "ensemble_weights_after": self.wma.get_weights(), "ensemble_mistakes_after": self.wma.get_mistake_counts(),
            "ensemble_epsilon": current_epsilon, "ensemble_processing_time_seconds": processing_time,
            "ensemble_status": "Completed" if final_sql and not final_sql.startswith("Error:") else "Failed"
        }

        logger.info(f"=== Finished Item {self.item_counter} (ID: {q_id}). Time: {processing_time:.2f}s ===")
        return result