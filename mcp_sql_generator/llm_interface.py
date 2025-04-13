# mcp_sql_generator/llm_interface.py
import time
import config
import os
import logging # Use logging instead of print for better control
from abc import ABC, abstractmethod
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import google.generativeai as genai
from together import Together
# from together import AuthenticationError # Removed this specific import

# --- Setup logger for this module ---
logger = logging.getLogger(__name__)

# --- Define placeholder for AuthenticationError if needed for isinstance ---
# Or just rely on checking error messages in _handle_error
try:
    from together.error import AuthenticationError
except ImportError:
    class AuthenticationError(Exception): pass
    # logger.debug("Specific 'together.error.AuthenticationError' not found.") # Optional debug msg


class LLMInterface(ABC):
    """Abstract Base Class for LLM interactions."""
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generates text based on the prompt."""
        pass

    def _handle_error(self, error: Exception, provider: str) -> str:
        """Handles common API errors."""
        error_type_name = type(error).__name__
        error_msg = str(error)
        # Use logger instead of print
        logger.error(f"Error during {provider} API call ({error_type_name}): {error_msg}", exc_info=True) # Add stack trace

        if isinstance(error, (RateLimitError, APITimeoutError)):
            logger.warning("Rate limit or timeout error. Consider increasing REQUEST_DELAY_SECONDS or checking API usage limits.")

        if provider == "TogetherAI":
            if "401" in error_msg or "authentication" in error_msg.lower() or isinstance(error, AuthenticationError):
                 logger.error(f"Potential Together AI Authentication Error: Check your API key (TOGETHER_API_KEY).")
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                 logger.warning(f"Potential Together AI Rate Limit Error.")
        elif provider == "Gemini":
             # Add specific Gemini error checks if needed
             pass

        return f"Error: Could not get response from {provider}. {error_type_name}"

# --- OpenAIClient Class (remains the same) ---
class OpenAIClient(LLMInterface):
    def __init__(self, api_key: str, model: str = config.DEFAULT_OPENAI_MODEL):
        if not api_key:
            raise ValueError("OpenAI API key not provided.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        logger.info(f"Initialized OpenAIClient with model: {self.model}") # Use logger

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL generation assistant following instructions precisely."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                stop=None,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                 return response.choices[0].message.content.strip()
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                logger.warning(f"OpenAI returned no choices or empty message. Finish reason: {finish_reason}")
                return f"Error: OpenAI returned empty response (finish reason: {finish_reason})."
        except Exception as e:
            return self._handle_error(e, "OpenAI")
        finally:
            time.sleep(config.REQUEST_DELAY_SECONDS)


# --- GeminiClient Class (Corrected generate method) ---
# --- GeminiClient Class (Corrected generate method) ---
class GeminiClient(LLMInterface):
    def __init__(self, api_key: str, model: str = config.DEFAULT_GEMINI_MODEL):
        # ... (init code remains the same) ...
        if not api_key:
            raise ValueError("Gemini API key not provided.")
        try:
             genai.configure(api_key=api_key)
             self.safety_settings = [
                 {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                 {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
             ]
             self.default_generation_config = genai.GenerationConfig(temperature=0.2)
             self.model_instance = genai.GenerativeModel(model_name=model)
             self.model = model
             logger.info(f"Initialized GeminiClient with model: {self.model}") # Use logger
        except Exception as e:
             logger.error(f"Error during Gemini initialization: {e}", exc_info=True) # Log stack trace
             raise

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """Generates text using Google Gemini with improved error handling."""
        current_config = genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        try:
            response = self.model_instance.generate_content(
                prompt,
                generation_config=current_config,
                safety_settings=self.safety_settings
            )

            # --- REMOVED problematic print statements ---
            # print(f"Gemini prompt: {prompt}") # Optional debug print - REMOVE or comment out
            # print(f"Gemini response: {response}") # Optional debug print - REMOVE or comment out
            # print(response.text) # <--- REMOVE THIS LINE (LINE 121)

            # --- Corrected Error Handling Logic ---
            # 1. Check for blocking FIRST
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason_str = str(response.prompt_feedback.block_reason)
                logger.warning(f"Gemini request blocked. Reason: {block_reason_str}")
                logger.debug(f"Blocked prompt (first 500 chars): {prompt[:500]}...")
                return f"Error: Gemini request blocked ({block_reason_str})."

            # 2. If not blocked, try accessing the text directly via response.text
            try:
                 if hasattr(response, 'text') and response.text is not None:
                     return response.text.strip()
                 else:
                    candidates_info = f"Candidates count: {len(response.candidates)}" if hasattr(response, 'candidates') else "Candidates attribute missing."
                    logger.warning(f"Gemini returned response with no text content. {candidates_info}")
                    # You might want to inspect the raw 'response' object here for more clues if needed
                    logger.debug(f"Raw Gemini response object when no text: {response}")
                    return "Error: Gemini returned empty response (no text)."

            except ValueError as e:
                 if "candidates" in str(e).lower() and "empty" in str(e).lower():
                      logger.warning("Gemini returned response with empty candidates list (caught by .text accessor).")
                      logger.debug(f"Raw Gemini response object on empty candidates: {response}")
                      return "Error: Gemini returned empty response (empty candidates)."
                 else:
                     logger.error(f"Unexpected ValueError accessing Gemini response.text: {e}", exc_info=True)
                     return self._handle_error(e, "Gemini")
            # --- End Corrected Error Handling ---

        except Exception as e:
            return self._handle_error(e, "Gemini")
        finally:
            time.sleep(config.REQUEST_DELAY_SECONDS)

# --- Rest of the llm_interface.py file (OpenAIClient, TogetherClient, get_llm_client) remains the same ---
# ... (Include the rest of the file as provided in the previous response) ...


# --- TogetherClient Class (remains the same as previous corrected version) ---
class TogetherClient(LLMInterface):
    def __init__(self, api_key: str | None, model: str = config.DEFAULT_TOGETHER_MODEL):
        resolved_api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not resolved_api_key:
            raise ValueError("Together AI API key not found in arguments or environment variable TOGETHER_API_KEY.")
        try:
             self.client = Together(api_key=resolved_api_key)
             self.model = model
             logger.info(f"Initialized TogetherClient with model: {self.model}") # Use logger
        except Exception as init_err:
             if "authentication" in str(init_err).lower() or "api key" in str(init_err).lower() or isinstance(init_err, AuthenticationError):
                 logger.error(f"Together AI Authentication Failed during initialization: {init_err}. Ensure TOGETHER_API_KEY is set correctly.", exc_info=True)
             else:
                 logger.error(f"Error initializing Together AI client: {init_err}", exc_info=True)
             raise init_err

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.0,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=False
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else "unknown"
                logger.warning(f"Together AI returned no choices or empty message. Finish reason: {finish_reason}")
                return f"Error: Together AI returned empty response (finish reason: {finish_reason})."
        except Exception as e:
            return self._handle_error(e, "TogetherAI")
        finally:
            time.sleep(config.REQUEST_DELAY_SECONDS)


# --- Factory Function (remains the same) ---
def get_llm_client(provider: str,
                   openai_api_key: str | None = None,
                   gemini_api_key: str | None = None,
                   together_api_key: str | None = None,
                   openai_model: str = config.DEFAULT_OPENAI_MODEL,
                   gemini_model: str = config.DEFAULT_GEMINI_MODEL,
                   together_model: str = config.DEFAULT_TOGETHER_MODEL
                   ) -> LLMInterface:
    provider_lower = provider.lower()
    if provider_lower == 'openai':
        if not openai_api_key: raise ValueError("OpenAI API key is required for OpenAI provider.")
        return OpenAIClient(api_key=openai_api_key, model=openai_model)
    elif provider_lower == 'gemini':
        if not gemini_api_key: raise ValueError("Gemini API key is required for Gemini provider.")
        return GeminiClient(api_key=gemini_api_key, model=gemini_model)
    elif provider_lower == 'togetherai':
        return TogetherClient(api_key=together_api_key, model=together_model)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Choose 'openai', 'gemini', or 'togetherai'.")