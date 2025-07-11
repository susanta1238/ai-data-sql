# integrations/openai/openai_client.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
# Add this line to help debug if this specific file version is being loaded.
# If this message doesn't appear in your terminal logs during startup or when hitting the endpoint,
# it means an older version of this file is being used due to caching or other environmental issue.
print("DEBUG: Loading integrations/openai/openai_client.py - Version Check 14") # Increment version for clarity
# --- END ADDITION ---


import logging
# Import asyncio for running the async test block
import asyncio
# Import inspect to check if methods are awaitable (used in pipeline debug)
import inspect
# Import sys for error handling in test block
import sys

# --- FIX: Import AsyncOpenAI ---
import openai # Keep import for openai.API*Error types
from openai import AsyncOpenAI # <-- Import AsyncOpenAI specifically


# Although pandas is not used directly in this file, keep the import commented out if other methods might use it later.
# import pandas as pd


# Import AppConfig class for type hinting
from utils.config_loader import AppConfig


# Get logger for this module AFTER the diagnostic print statement
logger = logging.getLogger(__name__)

class OpenAIClient:
    """
    Handles interactions with the OpenAI API for text generation.
    Uses asynchronous API calls via AsyncOpenAI client.
    """
    def __init__(self, config: AppConfig):
        """
        Initializes the client with application configuration.
        Sets the OpenAI API key and creates an asynchronous client instance.
        """
        # Basic type check for config object
        if not isinstance(config, AppConfig):
             raise TypeError("config must be an instance of AppConfig")

        openai_config = config.openai
        self.config = openai_config # Store the openai config dictionary

        api_key = openai_config.get('api_key')
        model = openai_config.get('model', 'gpt-4o-mini') # Default model if not in config
        timeout = openai_config.get('timeout', 30)      # Default timeout

        if not api_key:
            logger.error("OpenAI API key is not configured.")
            # In production, you might raise a ValueError here to prevent startup
            # raise ValueError("OpenAI API key is required.")

        # --- FIX: Use AsyncOpenAI to create an asynchronous client ---
        self.client = AsyncOpenAI( # <-- Use AsyncOpenAI here
            api_key=api_key,
            timeout=timeout
        )
        self.model = model
        # self.timeout = timeout # Stored via client instance now

        logger.info(f"OpenAIClient initialized with model '{self.model}' and timeout {timeout}s (Async).") # Log async init
        # --- ADD DEBUG LOG HERE ---
        # Log the type of the client instance created
        logger.debug(f"OpenAI client instance type: {type(self.client)}")
        # --- END DEBUG LOG ---
        if api_key:
             logger.info("OpenAI API key successfully loaded.")
        else:
             logger.warning("OpenAI API key is missing.")


    async def generate_text(self, prompt, max_tokens=500, temperature=None, custom_instructions=None):
        """
        Sends a text prompt to the OpenAI API for completion/chat response asynchronously.

        Args:
            prompt (str or list): The input prompt. Can be a simple string or
                                  a list of message objects for chat models.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Controls randomness. Overrides config if provided.
            custom_instructions (str): Optional system instructions for chat models.

        Returns:
            str: The generated text response, or None if an error occurred.
        """
        # Use the temperature from the config (available via self.config) as default if not provided in the call
        used_temperature = temperature if temperature is not None else self.config.get('temperature', 0.7)
        messages = []

        # Prepare messages list for the chat endpoint
        if custom_instructions:
             messages.append({"role": "system", "content": custom_instructions})
             logger.debug(f"Added system instruction: {custom_instructions[:100]}...")

        if isinstance(prompt, str):
            # If prompt is a simple string, add it as a user message
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            # If prompt is already a list, assume it's in the correct message format
            messages.extend(prompt) # Use extend to add to the list (which might have system message)
        else:
            logger.error("Invalid prompt format. Must be string or list of messages.")
            return None

        # Ensure there's at least one message
        if not messages:
             logger.error("No messages provided to generate text.")
             return None

        logger.debug(f"Sending prompt to OpenAI API asynchronously (model: {self.model}, temp: {used_temperature}, max_tokens: {max_tokens})...")

        try:
            # --- Await the create() method on the async client ---
            # This call should now return an awaitable object (ChatCompletion)
            # and correctly yield control until the API response is received.
            response = await self.client.chat.completions.create( # <--- Call create() with await on async client
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=used_temperature,
                # timeout is set on the client instance
                # Add other parameters as needed
            )

            if response.choices and response.choices[0].message.content is not None:
                generated_text = response.choices[0].message.content.strip()
                logger.debug(f"Received response from OpenAI API asynchronously. Content length: {len(generated_text)}")
                return generated_text
            else:
                logger.warning("OpenAI API returned no response or empty content.")
                logger.debug(f"OpenAI Response object: {response}")
                return None

        # Catch specific OpenAI API errors
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error (status {e.status_code}): {e.response.text}", exc_info=True)
            return None
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API connection error: {e}", exc_info=True)
            return None
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI API rate limit exceeded: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True)
            return None


    # You might add other methods later, defining them as 'async def' if they involve I/O (like API calls)


# --- Test Logic Function ---
async def test_openai_client():
    """
    Contains the test logic for the OpenAIClient.
    Designed to be called from the __main__ block using asyncio.run().
    """
    # Imports needed specifically for this test function's scope
    import sys
    try:
        from utils.logging_config import setup_logging
        from utils.config_loader import load_config
        # No explicit import of OpenAIClient needed here when running this file directly

    except ImportError as e:
        print(f"FATAL ERROR: Could not import core components for test: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Temporary Logging Setup for Test ---
    try:
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' }
                 self.app = {}
                 self.openai = {} # Include openai section for client initialization
                 # Add a dummy api_key for the test config if you don't rely on .env for the test run
                 # self.openai['api_key'] = 'dummy_key_for_test_config'
                 self.database = {}

        if not logging.getLogger().handlers:
            setup_logging(DummyConfigForLoggingTest())

        logger.info("Temporary logging setup complete for openai_client test function.")

    except Exception as e:
        print(f"Warning: Could not set up temporary logging in test function: {e}", file=sys.stderr)
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info("Using potentially basic logging for openai_client test function.")
    # --- End Temporary Logging Setup ---

    # --- Main Test Execution Block ---
    try:
        logger.info("--- Testing OpenAIClient ---")
        logger.debug("Successfully imported utilities for test function.")

        # Load configuration (This will load the API key from .env or environment)
        app_config = load_config()

        if not app_config.openai.get('api_key'):
             logger.error("OPENAI_API_KEY is not set. Skipping OpenAIClient test.")
             return # Exit the test function gracefully if no key

        # Initialize the client (now creates AsyncOpenAI instance)
        openai_client = OpenAIClient(app_config)

        # Check if the generate_text method is now awaitable (should be True)
        # This debug should now appear if the client initialized correctly
        logger.debug(f"Is generate_text awaitable? {inspect.iscoroutinefunction(openai_client.generate_text)}")


        # --- Await the generate_text calls ---
        # Test generating a simple conversational response asynchronously
        logger.info("Attempting to generate a simple conversational response asynchronously...")
        conversational_prompt = "Hello, what is your name?"
        # This await call should now work because generate_text is async and uses AsyncOpenAI client
        response_text = await openai_client.generate_text(conversational_prompt, max_tokens=50) # <-- AWAIT

        if response_text:
            logger.info(f"Conversational Response: {response_text}")
        else:
            logger.warning("Failed to get conversational response.")

        # Test generating a simple SQL-like response asynchronously
        logger.info("Attempting to generate a simple SQL query example asynchronously...")
        sql_prompt = "Generate a SQL query to select the first 10 rows from a table named 'Users'."
        # This await call should now work
        sql_response_text = await openai_client.generate_text(sql_prompt, max_tokens=100, temperature=0.1) # <-- AWAIT

        if sql_response_text:
             logger.info(f"SQL Example Response:\n{sql_response_text}")
        else:
             logger.warning("Failed to get SQL example response.")

        logger.info("--- OpenAIClient Test Complete ---")

    except Exception as e:
        logger.error(f"OpenAIClient test failed: An unexpected error occurred: {e}", exc_info=True)
        return

# --- Main execution block ---
if __name__ == "__main__":
    # This block runs when the file is executed directly.
    # We need to run the async test function using asyncio.run().
    try:
        # Use asyncio.run to execute the async test function
        asyncio.run(test_openai_client())
    except Exception as e:
         # Catch errors that happen even outside the test function, e.g., config loading failures
         print(f"FATAL ERROR during openai_client test main execution: {e}", file=sys.stderr)
         sys.exit(1)