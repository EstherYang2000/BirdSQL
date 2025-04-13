# mcp_sql_generator/agents/executor_agent.py
from .base_agent import BaseLLMAgent
from llm_interface import LLMInterface
from utils import parse_sql_from_response

class ExecutorAgent(BaseLLMAgent):
    def __init__(self, llm_client: LLMInterface):
        super().__init__(llm_client)

    def execute_plan(self, question: str, schema_string: str, evidence: str, plan: str) -> str:
        """Generates the initial SQL based strictly on the plan."""
        print("  [Executor Agent] Executing plan...")
        context = self._construct_context(question, schema_string, evidence)
        task = ("Based *strictly* on the Execution Plan provided below and the original context "
                "(Question, Schema, Evidence), write the SQLite SQL query. "
                "Output *only* the SQL query, nothing else.")
        additional_info = f"Execution Plan:\n{plan}"
        prompt = self._create_prompt(context, task, additional_info)
        raw_sql = self.llm_client.generate(prompt, max_tokens=500, temperature=0.0)
        sql = parse_sql_from_response(raw_sql)
        print(f"  [Executor Agent] Initial SQL generated: {sql[:100]}...")
        return sql