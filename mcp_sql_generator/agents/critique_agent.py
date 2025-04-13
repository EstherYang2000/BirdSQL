# mcp_sql_generator/agents/critique_agent.py
from .base_agent import BaseLLMAgent
from llm_interface import LLMInterface

class CritiqueAgent(BaseLLMAgent):
    def __init__(self, llm_client: LLMInterface):
        super().__init__(llm_client)

    def critique_sql(self, question: str, schema_string: str, evidence: str,
                       plan: str, current_sql: str, validation_result: tuple[bool, str | list | None]) -> str:
        """Generates a critique of the SQL query."""
        print("  [Critique Agent] Generating critique...")
        context = self._construct_context(question, schema_string, evidence)
        task = ("Evaluate the 'SQL Query to Critique' below based on the Plan, original context (Question, Schema, Evidence), "
                "and the Database Validation Attempt result. Identify potential errors, inconsistencies, or areas for improvement. Consider: \n"
                "1. Correctness: Does it retrieve the intended data according to the question and plan?\n"
                "2. Schema Adherence: Are table/column names valid? Are quotes needed/used correctly?\n"
                "3. Plan Following: Does it implement the plan accurately?\n"
                "4. Completeness: Does it address all parts of the question?\n"
                "5. Syntax/Execution: Did it pass validation? If not, what was the error? (See 'Validation Attempt' below)\n"
                "Provide a concise critique. If the query looks correct and passed validation, state that clearly.")

        validation_success, validation_output = validation_result
        validation_info = f"Validation Attempt Result:\n"
        if validation_success:
            validation_info += "  - Success: True\n"
            if validation_output is not None: # Can be None for non-SELECT or empty result
                validation_info += f"  - Execution Output (Sample): {str(validation_output)[:200]}...\n" # Show sample output
            else:
                 validation_info += f"  - Execution Output: Query executed successfully (non-SELECT or empty result).\n"
        else:
            validation_info += "  - Success: False\n"
            validation_info += f"  - Error Message: {validation_output}\n"


        additional_info = (f"Plan:\n{plan}\n\n"
                           f"{validation_info}\n" # Include validation result
                           f"SQL Query to Critique:\n```sql\n{current_sql}\n```")

        prompt = self._create_prompt(context, task, additional_info)
        critique = self.llm_client.generate(prompt, max_tokens=500, temperature=0.3) # Increased tokens slightly
        print("  [Critique Agent] Critique received.")
        return critique