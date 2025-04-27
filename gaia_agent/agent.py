import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from gaia_agent.tools import tool_list
from gaia_agent.prompts import general_instructions
from dotenv import load_dotenv
import io
import sys

load_dotenv()


class GAIAAgent:
    """Agent for the GAIA challenge."""

    def __init__(self, config: dict):
        provider = config.get("provider", "")
        model = config.get("model_name", "")
        temperature = config.get("temperature", None)

        if provider == "gemini":
            llm = GoogleGenAI(
                model=model,
                api_key=os.environ.get("GEMINI_API_KEY"),
                temperature=temperature,
            )
        elif provider == "openai":
            llm = OpenAI(
                model=model,
                api_key=os.environ.get("OPENAI_API_KEY"),
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.agent = ReActAgent(
            tools=tool_list,
            llm=llm,
            verbose=True,
            memory=None,
        )

    async def run(self, question: str, return_reasoning_process: bool = False):
        """Run the agent with the given task."""
        prompt = general_instructions.format(question=question)
        if return_reasoning_process:
            # Capture the reasoning process being printed to stdout
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            try:
                # Let the agent reason
                response = await self.agent.achat(prompt)
            finally:
                # Restore stdout
                sys.stdout = old_stdout
            captured_output = new_stdout.getvalue()
            return response.response, captured_output
        else:
            # Let the agent reason
            response = await self.agent.achat(prompt)
            return response.response, ""
