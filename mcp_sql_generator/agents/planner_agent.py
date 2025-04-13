# mcp_sql_generator/agents/planner_agent.py
from .base_agent import BaseLLMAgent
from llm_interface import LLMInterface

class PlannerAgent(BaseLLMAgent):
    def __init__(self, llm_client: LLMInterface):
        super().__init__(llm_client)

    def generate_plan(self, question: str, schema_string: str, evidence: str) -> str:
        """Generates the SQL generation plan."""
        print("  [Planner Agent] Generating plan...")
        context = self._construct_context(question, schema_string, evidence)
        task = ("Analyze the User Question, Database Schema, and Provided Evidence. "
                "Create a detailed, step-by-step plan to construct the SQLite SQL query "
                "that answers the question. Focus on: necessary tables, columns, JOINs, "
                "filtering (WHERE), aggregation/grouping (GROUP BY), ordering (ORDER BY), "
                "and limits (LIMIT). Explain the reasoning for each step clearly.")
        prompt = self._create_prompt(context, task)
        plan = self.llm_client.generate(prompt, max_tokens=700, temperature=0.1)
        print("  [Planner Agent] Plan generated.")
        return plan