# mcp_sql_generator/wma.py
import math
import random
import logging

# Configure logger for this module
logger = logging.getLogger(__name__)
# Basic configuration if run standalone or not configured by main app
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def auto_select_epsilon(num_experts, num_rounds=None, best_mistakes=None):
    """
    Auto-calculate epsilon based on theory or heuristics.
    """
    if num_experts < 1:
        raise ValueError("num_experts must be >= 1")

    log_N = math.log(num_experts) if num_experts > 1 else 0 # Avoid log(1) = 0 denominator issues, handle num_experts=1

    # Prefer best_mistakes if available and valid
    if best_mistakes is not None and best_mistakes > 0:
        # Add small constant to avoid division by zero if best_mistakes is somehow zero initially
        epsilon = math.sqrt(log_N / (best_mistakes + 1e-6))
        logger.debug(f"Calculated epsilon based on best_mistakes ({best_mistakes}): {epsilon}")
        return epsilon
    # Fallback to num_rounds
    elif num_rounds is not None and num_rounds > 0:
        epsilon = math.sqrt(log_N / num_rounds)
        logger.debug(f"Calculated epsilon based on num_rounds ({num_rounds}): {epsilon}")
        return epsilon
    # Fallback to a default if neither is provided (or handle error)
    else:
        default_epsilon = 0.1 # Or some other reasonable default
        logger.warning(f"Insufficient info for auto_select_epsilon. Using default: {default_epsilon}")
        # Or raise ValueError("Need num_rounds or best_mistakes for auto_select_epsilon")
        return default_epsilon

class WeightedMajorityAlgorithm:
    """
    Implements WMA (Weighted Majority Algorithm).
    Experts maintain weights, penalized for incorrect predictions.
    """

    def __init__(self, experts_config: list[dict] | None = None, epsilon: float = 0.1, initial_weight: float = 1.0):
        """
        Initialize WMA.

        Args:
            experts_config (list[dict] | None): List of expert config dicts, e.g., [{'name': 'expert1'}, ...].
                                               Weights initialized if provided.
            epsilon (float): Learning rate / penalty factor (0 < epsilon < 1).
            initial_weight (float): Starting weight for all experts.
        """
        self.experts = {}  # {expert_name: weight}
        self.mistake_counter = {} # {expert_name: count}
        self.epsilon = epsilon
        self.initial_weight = initial_weight

        if experts_config:
            for expert_conf in experts_config:
                name = expert_conf.get("name")
                if name:
                    self.add_expert(name, self.initial_weight)
                else:
                    logger.warning("Expert config found without a 'name' field.")

        logger.info(f"WMA initialized with epsilon={self.epsilon:.4f}. Experts: {list(self.experts.keys())}")

    def add_expert(self, expert_name: str, init_weight: float | None = None):
        """Add a new expert or reset an existing one."""
        weight = init_weight if init_weight is not None else self.initial_weight
        if expert_name not in self.experts:
            self.experts[expert_name] = weight
            self.mistake_counter[expert_name] = 0
            logger.debug(f"Added expert '{expert_name}' with weight {weight:.4f}")
        else:
            self.experts[expert_name] = weight # Allow resetting weight
            self.mistake_counter[expert_name] = 0 # Reset mistakes on re-add/reset
            logger.debug(f"Reset expert '{expert_name}' with weight {weight:.4f}")


    def update_weights(self, expert_contributions: dict[str, bool], strategy: str):
        """
        Update expert weights based on whether they contributed *any* correct prediction
        in the current round (lenient update).

        Args:
            expert_contributions (dict[str, bool]): Map of expert_name -> bool (True if contributed a correct SQL).
            strategy (str): The voting strategy ('wma', 'rwma', 'naive'). 'naive' skips updates.
        """
        if strategy == "naive":
            # Only update mistake counts in naive mode if desired (e.g., for tracking)
            for expert_name, was_correct_contributor in expert_contributions.items():
                 if expert_name in self.mistake_counter and not was_correct_contributor:
                      self.mistake_counter[expert_name] += 1
            return # No weight updates for naive strategy

        updated_experts = []
        for expert_name, was_correct_contributor in expert_contributions.items():
            if expert_name not in self.experts:
                logger.warning(f"Attempted to update weight for unknown expert '{expert_name}'")
                continue

            if not was_correct_contributor:
                old_weight = self.experts[expert_name]
                # Ensure epsilon is less than 1 to avoid negative weights
                penalty_factor = max(0, 1 - self.epsilon)
                new_weight = old_weight * penalty_factor
                self.experts[expert_name] = new_weight
                self.mistake_counter[expert_name] += 1
                updated_experts.append(f"{expert_name}({new_weight:.3f})")
            # else: weight remains unchanged if they contributed a correct answer

        if updated_experts:
             logger.debug(f"Weights updated (decreased) for: {', '.join(updated_experts)}")
        else:
             logger.debug("No weights decreased in this round.")


    def get_mistake_counts(self) -> dict[str, int]:
        """Return a copy of the mistake counts for all experts."""
        return self.mistake_counter.copy()

    def _perform_vote(self, predictions_dict: dict[str, list[str]]) -> tuple[str | None, list[str], float]:
        """Internal helper for the weighted majority voting logic."""
        sql_to_weight = {}
        sql_to_experts = {}

        if not predictions_dict:
            logger.error("No SQL predictions received for voting.")
            return None, [], 0.0

        valid_predictions_found = False
        for expert_name, sql_list in predictions_dict.items():
            # Ensure expert exists and handle potential empty lists
            if expert_name not in self.experts or not sql_list:
                # logger.warning(f"Expert '{expert_name}' not in WMA list or provided no predictions.")
                continue

            expert_weight = self.experts[expert_name]
            if expert_weight <= 0: # Skip experts with zero or negative weight
                 continue

            for sql_str in sql_list:
                 # Basic check to avoid voting on obvious errors if possible
                 if not sql_str or sql_str.startswith("Error:") or len(sql_str) < 5:
                     continue # Skip potentially invalid entries

                 valid_predictions_found = True
                 sql_key = sql_str.strip().upper() # Normalize SQL for voting consistency
                 if sql_key not in sql_to_weight:
                     sql_to_weight[sql_key] = 0.0
                     sql_to_experts[sql_key] = []
                 sql_to_weight[sql_key] += expert_weight
                 # Store original expert name associated with this normalized key
                 if expert_name not in sql_to_experts[sql_key]:
                    sql_to_experts[sql_key].append(expert_name)


        if not valid_predictions_found or not sql_to_weight:
            logger.error("No valid, non-empty SQL predictions found from active experts for voting.")
            return None, [], 0.0

        # Find the SQL key (normalized) with the highest score
        best_sql_key = max(sql_to_weight, key=sql_to_weight.get)
        best_weight = sql_to_weight[best_sql_key]
        chosen_experts = sql_to_experts[best_sql_key] # Experts who proposed this winning SQL

        # Find the original (non-normalized) SQL string corresponding to the best key
        # This assumes the first occurrence is representative enough
        original_sql = None
        for sql_list in predictions_dict.values():
             for sql_str in sql_list:
                 if sql_str and sql_str.strip().upper() == best_sql_key:
                     original_sql = sql_str # Found an original version
                     break
             if original_sql:
                 break

        if original_sql is None: # Fallback if somehow no original is found (shouldn't happen)
             logger.error(f"Could not find original SQL for winning key '{best_sql_key}'. Returning key.")
             original_sql = best_sql_key


        logger.debug(f"WMA Vote Result: SQL='{original_sql[:100]}...', Weight={best_weight:.4f}, Experts={chosen_experts}")
        return original_sql, chosen_experts, best_weight


    def weighted_majority_vote(self, predictions_dict: dict[str, list[str]]) -> tuple[str | None, list[str], float]:
        """Perform a deterministic weighted majority vote."""
        return self._perform_vote(predictions_dict)


    def randomized_weighted_majority_vote(self, predictions_dict: dict[str, list[str]]) -> tuple[str | None, list[str], float]:
        """Perform a randomized weighted majority vote based on expert weights."""
        if not predictions_dict:
            logger.error("No SQL predictions received for RWMA.")
            return None, [], 0.0

        # Filter experts with positive weight and valid predictions
        active_experts = {
            name: self.experts[name]
            for name, preds in predictions_dict.items()
            if name in self.experts and self.experts[name] > 0 and preds and any(p and not p.startswith("Error:") for p in preds)
        }

        if not active_experts:
             logger.error("No active experts with positive weight and valid predictions for RWMA.")
             # Fallback: maybe try a simple majority vote? Or return error.
             return self._perform_vote(predictions_dict) # Try WMA as fallback

        total_weight = sum(active_experts.values())

        # Select an expert based on normalized weights
        try:
             selected_expert = random.choices(
                 population=list(active_experts.keys()),
                 weights=list(active_experts.values()),
                 k=1
             )[0]
        except Exception as e:
             logger.error(f"Error during RWMA expert selection: {e}. Falling back to WMA.")
             return self._perform_vote(predictions_dict) # Fallback

        # Select a SQL from the chosen expert's list
        valid_sql_list = [p for p in predictions_dict[selected_expert] if p and not p.startswith("Error:")]
        if not valid_sql_list:
             logger.warning(f"RWMA selected expert '{selected_expert}' but they had no valid SQL. Falling back to WMA.")
             return self._perform_vote(predictions_dict) # Fallback

        final_sql = random.choice(valid_sql_list)
        final_weight = active_experts[selected_expert] # Weight of the chosen expert
        logger.debug(f"RWMA Vote Result: Chosen Expert='{selected_expert}', SQL='{final_sql[:100]}...', Expert Weight={final_weight:.4f}")

        return final_sql, [selected_expert], final_weight


    def naive_vote(self, predictions_dict: dict[str, list[str]]) -> tuple[str | None, list[str], float]:
        """Perform a simple majority vote (ties broken arbitrarily)."""
        # Essentially WMA with all weights = 1.0
        temp_experts = {name: 1.0 for name in predictions_dict if predictions_dict[name]}
        if not temp_experts:
             logger.error("No valid predictions for naive vote.")
             return None, [], 0.0

        original_weights = self.experts # Backup original weights
        self.experts = temp_experts # Temporarily set all weights to 1 for voting
        result_sql, result_experts, _ = self._perform_vote(predictions_dict)
        self.experts = original_weights # Restore original weights

        # The returned 'weight' isn't meaningful here, return count maybe?
        vote_count = len(result_experts) if result_sql else 0
        logger.debug(f"Naive Vote Result: SQL='{result_sql[:100]}...', Count={vote_count}, Experts={result_experts}")
        return result_sql, result_experts, float(vote_count)

    def get_weights(self) -> dict[str, float]:
        """Return a copy of the current expert weights."""
        return self.experts.copy()

    def set_epsilon(self, epsilon: float):
        """Update the epsilon value."""
        if 0 < epsilon < 1:
             self.epsilon = epsilon
             logger.info(f"WMA epsilon updated to {self.epsilon:.4f}")
        else:
             logger.warning(f"Invalid epsilon value: {epsilon}. Must be between 0 and 1.")