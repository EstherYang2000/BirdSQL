# mcp_sql_generator/agents/sql_generator_agent.py
from .base_agent import BaseLLMAgent
from llm_interface import LLMInterface
from utils import parse_sql_from_response

class SQLGeneratorAgent(BaseLLMAgent):
    """
    A simplified agent focused on generating a single SQL candidate.
    Combines planning and execution implicitly in the prompt.
    """
    def __init__(self, llm_client: LLMInterface):
        super().__init__(llm_client)

    def generate_sql(self, question: str, schema_string: str, evidence: str) -> str:
        """Generates a single SQL candidate."""
        # print(f"  [SQLGeneratorAgent ({self.llm_client.model})] Generating candidate...") # Optional: Log model
        context = self._construct_context(question, schema_string, evidence)
        # Task description asks for direct SQL generation
        task = ("Analyze the User Question, Database Schema, and any Provided Evidence. "
                "Your goal is to produce the correct SQLite SQL query that answers the question. "
                "Consider necessary tables, columns, JOINs, filtering (WHERE), aggregation/grouping (GROUP BY), "
                "ordering (ORDER BY), and limits (LIMIT). "
                "Output *only* the final SQLite SQL query, with no explanation or commentary.")

        prompt = self._create_prompt(context, task)
        # Use parameters suitable for direct generation (low temperature)
        raw_sql = self.llm_client.generate(prompt, max_tokens=500, temperature=0.05)
        sql_candidate = parse_sql_from_response(raw_sql)
        # print(f"  [SQLGeneratorAgent ({self.llm_client.model})] Candidate: {sql_candidate[:100]}...") # Optional: Log model
        return sql_candidate