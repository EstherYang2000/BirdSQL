# mcp_sql_generator/agents/base_agent.py
from llm_interface import LLMInterface

class BaseLLMAgent:
    def __init__(self, llm_client: LLMInterface):
        self.llm_client = llm_client

    def _construct_context(self, question: str, schema_string: str, evidence: str) -> str:
        """Creates the common context part (Schema, Evidence, Question)."""
        context = f"### Database Schema:\n{schema_string}\n\n"
        if evidence and isinstance(evidence, str) and evidence.strip():
            clean_evidence = evidence.replace("/*Here are some data information about database references.\n /*Evidence:","").replace("\n","").strip()
            if clean_evidence:
                context += f"### Provided Evidence/Context:\n{clean_evidence}\n\n"
        context += f"### User Question:\n{question}\n"
        return context

    def _create_prompt(self, context: str, task_description: str, additional_info: str = "") -> str:
        """Helper to assemble the final prompt for the LLM."""
        prompt = f"{context}\n"
        prompt += f"---\n"
        prompt += f"### Agent Task:\n{task_description}\n"
        if additional_info:
            prompt += f"\n{additional_info}\n"
        prompt += f"---\n"
        prompt += f"### Output:"
        return prompt