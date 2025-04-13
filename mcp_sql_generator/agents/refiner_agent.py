# mcp_sql_generator/agents/refiner_agent.py
from .base_agent import BaseLLMAgent
from llm_interface import LLMInterface
from utils import parse_sql_from_response

class RefinerAgent(BaseLLMAgent):
    def __init__(self, llm_client: LLMInterface):
        super().__init__(llm_client)

    def refine_sql(self, question: str, schema_string: str, evidence: str,
                     plan: str, current_sql: str, critique: str) -> str:
        """Refines the SQL based on the critique."""
        print("  [Refiner Agent] Refining SQL...")
        context = self._construct_context(question, schema_string, evidence)
        task = ("Based *only* on the Critique provided below, revise the 'SQL Query to Refine' to address the identified issues. "
                "Adhere to the original Plan and context (Question, Schema, Evidence). "
                "Focus on fixing errors mentioned in the critique (especially syntax/execution errors if present). "
                "Output *only* the improved SQLite SQL query, nothing else.")
        additional_info = (f"Original Plan:\n{plan}\n\n"
                           f"SQL Query to Refine:\n```sql\n{current_sql}\n```\n\n"
                           f"Critique:\n{critique}")
        prompt = self._create_prompt(context, task, additional_info)
        raw_refined_sql = self.llm_client.generate(prompt, max_tokens=500, temperature=0.0)
        refined_sql = parse_sql_from_response(raw_refined_sql)
        print(f"  [Refiner Agent] Refined SQL generated: {refined_sql[:100]}...")
        return refined_sql