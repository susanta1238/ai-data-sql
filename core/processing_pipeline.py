# core/processing_pipeline.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
print("DEBUG: Loading core/processing_pipeline.py - Version Check 13") # Increment version
# --- END ADDITION ---


import logging
import inspect # Import inspect module at the top
import pandas as pd # Needed for DataFrame check/handling

# Import necessary components from your services
from utils.config_loader import AppConfig
from integrations.database.db_connector import DatabaseConnector
from integrations.database.schema_loader import SchemaLoader
# Import OpenAIClient class (its generate_text method is async)
from integrations.openai.openai_client import OpenAIClient
from integrations.database.sql_safety import SQLSafetyChecker
# Import QueryExecutor class (RuntimeError is a built-in exception, not imported here)
from integrations.database.query_executor import QueryExecutor # Import QueryExecutor class

# Import the built-in RuntimeError for catching execution errors
from builtins import RuntimeError # Explicitly import built-in exception

# --- Import ConversationManager ---
from .conversation_manager import ConversationManager # Using relative import
from typing import List, Dict, Any, Optional # Import typing for history hints


logger = logging.getLogger(__name__) # Get logger at module level

class ConversationPipeline:
    """
    Orchestrates the process of receiving a user message, classifying intent,
    generating SQL (if needed), validating/executing SQL, and returning a response.
    """
    # --- __init__ method that takes arguments ---
    def __init__(
        self,
        app_config: AppConfig,
        db_connector: DatabaseConnector,
        schema_loader: SchemaLoader,
        openai_client: OpenAIClient | None, # OpenAI is optional if key is missing
        sql_safety_checker: SQLSafetyChecker | None, # Safety checker is optional if init failed
        # --- Add conversation_manager dependency ---
        conv_manager: ConversationManager, # Accept ConversationManager instance
    ):
        """
        Initializes the pipeline with instances of all required services.
        """
        # Basic type/None checks for dependencies
        if not isinstance(app_config, AppConfig):
             raise TypeError("app_config must be an instance of AppConfig")
        if not isinstance(db_connector, DatabaseConnector):
             raise TypeError("db_connector must be an instance of DatabaseConnector")
        if not isinstance(schema_loader, SchemaLoader):
             raise TypeError("schema_loader must be an instance of SchemaLoader")
        # Check if ConversationManager instance is provided
        if not isinstance(conv_manager, ConversationManager):
             raise TypeError("conv_manager must be an instance of ConversationManager")
        # Note: openai_client and sql_safety_checker can be None


        self.app_config = app_config
        self.db_connector = db_connector
        self.schema_loader = schema_loader
        # --- CRITICAL: Ensure openai_client is assigned here ---
        # This was the source of the AttributeError before.
        self.openai_client = openai_client # <--- Make SURE this line exists and is not commented out

        self.sql_safety_checker = sql_safety_checker # Can be None
        # --- Store ConversationManager instance ---
        self.conv_manager = conv_manager


        # Initialize QueryExecutor here, as it's a dependency of the pipeline logic
        # It needs the db_connector.
        if self.db_connector:
             self.query_executor = QueryExecutor(self.db_connector)
        else:
             # Query executor cannot be initialized if DB connection failed
             self.query_executor = None
             logger.warning("QueryExecutor not initialized due to missing DatabaseConnector.")


        # Cache schema if needed, or rely on schema_loader to fetch/cache internally
        # For simplicity, we'll load schema when needed in process_message for now.


        logger.info("ConversationPipeline initialized.")

    # --- process_message method uses ConversationManager ---
    async def process_message(self, user_message: str, conversation_id: str | None):
        """
        Processes a single user message asynchronously, using conversation history.

        Args:
            user_message (str): The natural language message from the user.
            conversation_id (str | None): ID to track conversation history. Required for history.

        Returns:
            dict: A dictionary containing the response type ('data' or 'text')
                  and the content (DataFrame or string message).
                  Example: {'type': 'data', 'content': df, 'generated_sql': sql}
                           {'type': 'text', 'content': 'Hello!'}

        Raises:
             ValueError: If generated SQL is unsafe.
             RuntimeError: If database query execution fails.
             Exception: For other unexpected errors.
        """
        logger.info(f"Processing message: '{user_message}' for conversation {conversation_id}")

        # --- Retrieve conversation history using conv_manager ---
        conversation_history = []
        if conversation_id:
            # Get history from the manager instance
            conversation_history = self.conv_manager.get_history(conversation_id)
            logger.debug(f"Retrieved {len(conversation_history)} messages for conversation {conversation_id}.")
        else:
            # This warning means the API endpoint didn't provide an ID
            logger.warning("No conversation_id provided to process_message. History will not be used or stored.")


        # --- Check if critical services are available ---
        # OpenAI client is needed for classification and potentially conversational responses
        # This is the line that was throwing the AttributeError before, now fixed by correct __init__
        if self.openai_client is None: # <--- This check should now find the self.openai_client attribute
             logger.error("OpenAI client is not available. Cannot classify intent or generate responses.")
             # Do NOT add to history if critical service is down before processing (as history isn't integrated yet)
             return {'type': 'text', 'content': "Sorry, the AI service is currently unavailable."}


        # --- 1. Classify Intent ---
        # Use OpenAIClient to determine if user wants data or conversation
        # Pass user message and conversation history to the classifier
        # This await call is now correct because _classify_intent is async
        intent = await self._classify_intent(user_message, conversation_history) # Pass history to classifier

        logger.info(f"Message classified with intent: {intent}")

        # --- 2. Route Based on Intent ---
        response = None # Initialize response variable
        generated_sql = None # Initialize generated_sql variable

        try: # Wrap the main processing logic in try/except to ensure history is added on success
            if intent == 'DATA_QUERY':
                logger.info("Intent is DATA_QUERY. Proceeding with SQL generation.")

                # --- Check if critical services for data query are available ---
                # Database, Schema, Executor, Safety Checker are all needed for data queries
                if self.db_connector is None or self.schema_loader is None or self.query_executor is None or self.sql_safety_checker is None:
                     logger.error("Critical services for data query are unavailable (DB, Schema, Executor, Safety Checker).")
                     response = {'type': 'text', 'content': "Sorry, I'm unable to access the database to answer your question right now."}
                else:
                     # TODO: Implement DATA_QUERY intent logic

                     # 2a. Load Database Schema (or use cached)
                     try:
                         schema_info = self.schema_loader.get_full_schema()
                         logger.debug(f"Loaded schema for {len(schema_info)} tables.")
                         if not schema_info:
                              logger.warning("Loaded schema is empty. Cannot generate meaningful queries.")
                              response = {'type': 'text', 'content': "Sorry, I was able to connect to the database, but it appears to be empty or I couldn't load the schema correctly."}

                     except Exception as e:
                          logger.error(f"Failed to load database schema during query processing: {e}", exc_info=True)
                          response = {'type': 'text', 'content': "Sorry, I'm having trouble accessing the database schema right now. Please try again later."}

                     # Only proceed if schema loading was successful
                     if response is None:
                         # 2b. Generate SQL using OpenAIClient
                         try:
                             # Pass user message, schema, and conversation history to the SQL generation method
                             generated_sql = await self._generate_sql(user_message, schema_info, conversation_history) # Pass history
                             logger.info(f"Generated SQL: {generated_sql}")
                         except Exception as e:
                              logger.error(f"Failed to generate SQL using OpenAI: {e}", exc_info=True)
                              response = {'type': 'text', 'content': "Sorry, I'm having trouble generating a database query right now. Please try again later."}

                         # Only proceed if SQL generation was successful
                         if response is None:
                             # 2c. Validate Generated SQL using SQLSafetyChecker
                             is_safe = self.sql_safety_checker.is_safe(generated_sql, schema_info)

                             if not is_safe:
                                 logger.warning(f"Generated SQL deemed unsafe: {generated_sql}")
                                 # Raising a ValueError here will be caught by the API endpoint's try/except
                                 raise ValueError("The generated query was deemed unsafe.")
                                 # Alternative: response = {'type': 'text', 'content': "Sorry, I cannot execute that query for security reasons."}

                             # 2d. Execute Validated SQL using QueryExecutor
                             try:
                                 results = self.query_executor.execute_select(generated_sql) # Call SYNCHRONOUSLY

                                 logger.info(f"Query executed successfully. Results: {type(results)}")

                                 # 2e. Format Results (DataFrame or Message) and set response
                                 response = {'type': 'data', 'content': results, 'generated_sql': generated_sql}

                             # --- Catch specific exceptions raised by the executor ---
                             except RuntimeError as e:
                                 logger.error(f"Database query execution failed in pipeline: {e}")
                                 # Re-raise the RuntimeError. The API endpoint will catch this specific type.
                                 raise e # Re-raise the RuntimeError


            elif intent == 'CONVERSATIONAL':
                logger.info("Intent is CONVERSATIONAL. Proceeding with text response.")
                 # OpenAI client availability is checked at the start of process_message

                # TODO: Handle CONVERSATIONAL intent logic

                # Generate a conversational response using OpenAIClient
                # Pass user message and history to generate text
                # Requires self.openai_client (checked earlier)
                # _generate_conversational_response is async and awaits openai_client.generate_text
                try:
                    conversational_response = await self._generate_conversational_response(user_message, conversation_history) # Pass history
                    logger.debug(f"Generated conversational response: {conversational_response}")
                    response = {'type': 'text', 'content': conversational_response}
                except Exception as e:
                    logger.error(f"Failed to generate conversational response using OpenAI: {e}", exc_info=True)
                    response = {'type': 'text', 'content': "Sorry, I'm having trouble responding right now."}


            elif intent == 'SCHEMA_HELP':
                 logger.info("Intent is SCHEMA_HELP.")
                 # TODO: Implement SCHEMA_HELP intent logic

                 # Schema loader availability checked implicitly if db_connector is checked
                 if self.schema_loader is None:
                     logger.error("SchemaLoader is not available. Cannot provide schema help.")
                     response = {'type': 'text', 'content': "Sorry, I'm unable to provide details about the database schema because the schema loader is not available."}
                 else:
                     try:
                         # Load schema if not cached
                         schema_info = self.schema_loader.get_full_schema()
                         if not schema_info:
                              logger.warning("Schema help requested, but loaded schema is empty.")
                              response = {'type': 'text', 'content': "I was able to connect to the database, but it appears to be empty or I couldn't load the schema correctly."}
                         else:
                             # Format the schema information into a user-friendly text
                             schema_text = self._format_schema_for_user(schema_info) # Implement formatting
                             logger.debug("Formatted schema for user.")
                             response = {'type': 'text', 'content': schema_text}
                     except Exception as e:
                          logger.error(f"Failed to provide schema help: {e}", exc_info=True)
                          response = {'type': 'text', 'content': "Sorry, an unexpected error occurred while trying to get schema details."}


            elif intent == 'OTHER' or intent == 'UNCLEAR':
                logger.info("Intent is OTHER/UNCLEAR.")
                # TODO: Implement OTHER/UNCLEAR intent logic

                # Return a canned response or a general AI response if appropriate
                # Can optionally use OpenAI for a more general response here if needed (but check openai_client availability)
                response = {'type': 'text', 'content': "I can help you with questions about our data. What information are you looking for?"}

            else:
                logger.error(f"Unknown intent returned by classifier: {intent}")
                response = {'type': 'text', 'content': "Sorry, I didn't understand that."}

        except (ValueError, RuntimeError, Exception) as e: # Catch errors during processing
             # Specific ValueErrors and RuntimeErrors raised within the logic will be caught here
             # Generic exceptions too.
             # We need to ensure we log this error, but still try to add the turn to history
             # before re-raising it to be caught by the API endpoint.
             logger.error(f"Error during processing message in pipeline: {e}", exc_info=True)
             # Ensure response is set to a default error message if it wasn't already
             if response is None:
                  # Use the exception message as the response content in case of error
                  response = {'type': 'text', 'content': f"An error occurred: {e}"} # Use error message (sanitize in prod)

             # The exception will be re-raised below after adding to history

        # --- Add current turn to history AFTER processing, but before returning ---
        # This happens whether an error occurred or processing was successful.
        # Only add to history if conversation_id was provided
        # Uses the conv_manager instance stored in __init__
        if conversation_id is not None and self.conv_manager is not None:
             # Add the user message and the final response content to history
             # Handle cases where the response is None or an error occurred
             final_response_content_for_history = response.get('content') if response and 'content' in response else str(e) if 'e' in locals() else "Unknown error."
             # Handle DataFrame content for history storage (store string representation)
             if isinstance(final_response_content_for_history, pd.DataFrame):
                 final_response_content_for_history = final_response_content_for_history.head().to_string() # Store string representation


             try:
                 self.conv_manager.add_turn(conversation_id, user_message, final_response_content_for_history)
                 logger.debug(f"Added user message and assistant response/error to history for {conversation_id}.")
             except Exception as history_e:
                  logger.error(f"Failed to add turn to history for {conversation_id}: {history_e}", exc_info=True)
        elif conversation_id is None and response is not None:
             # Only log this warning if there was a response but no ID
             logger.warning("Conversation ID is None, skipping adding turn to history.")


        # Re-raise the exception if one was caught, so the API endpoint can handle it
        # Check if an exception variable 'e' is active from the except block before re-raising
        if 'e' in locals() and isinstance(e, Exception):
             # Re-raise the exception that was caught
             # If it was a ValueError or RuntimeError from earlier, re-raise that specific type
             # Otherwise, re-raise as a generic Exception (or the caught Exception instance)
             if isinstance(e, (ValueError, RuntimeError)):
                 raise e
             else:
                 # Re-raise the unhandled exception
                 raise Exception("An unexpected error occurred during processing.") from e


        # Ensure a response was generated before returning
        if response is None:
             # If this point is reached without a response or exception, it indicates a logic error.
             logger.error("Processing pipeline finished without generating a response and no exception was caught.")
             response = {'type': 'text', 'content': "An unhandled logic error occurred while processing."}

        # Include generated_sql in the response dict if available (useful for debugging)
        # Check if generated_sql was set during the DATA_QUERY path and is not already in response
        if generated_sql is not None and response is not None and 'generated_sql' not in response:
             response['generated_sql'] = generated_sql


        return response # Return the final response dictionary


    # --- Private Helper Methods (Implementing with LLM calls) ---

    # --- _classify_intent method uses conversation_history ---
    async def _classify_intent(self, user_message: str, conversation_history: List[Dict[str, str]]):
        """
        Uses the OpenAIClient to classify the user's intent asynchronously.
        Includes conversation history in the prompt.
        Possible return values: 'DATA_QUERY', 'CONVERSATIONAL', 'SCHEMA_HELP', 'OTHER', 'UNCLEAR'.
        Requires self.openai_client to be available.
        """
        logger.debug(f"Classifying intent for message: '{user_message}'")

        if self.openai_client is None:
             logger.error("Attempted intent classification but OpenAI client is not available.")
             return 'OTHER' # Fallback if client is None

        # --- DEBUG LOGS BEFORE AWAIT ---
        # These logs should now execute correctly before the problematic await call
        logger.debug(f"OpenAI client instance: {self.openai_client}")
        # Check if generate_text is a coroutine function (should now be True)
        logger.debug(f"Is generate_text a coroutine function? {inspect.iscoroutinefunction(self.openai_client.generate_text)}")
        # --- END DEBUG LOGS ---


        system_instructions = """You are an intent classifier for a database AI assistant.
Your task is to analyze user queries and categorize them into one of the following intents based on the current message and conversation history:

- 'DATA_QUERY': The user is asking for specific data FROM the database. Look for keywords like "show", "get", "list", "count", "total", "average", "find", "what is", "how many", or follow-up questions related to previous data queries (e.g., "Now show me the ones from California").
- 'CONVERSATIONAL': The user is engaging in general chat, greetings, or non-data related questions ABOUT the domain or the assistant itself. Look for keywords like "hello", "hi", "how are you", "tell me about [topic]", "thank you", "explain".
- 'SCHEMA_HELP': The user is asking about the STRUCTURE of the database (tables, columns, data types, what data is available). Look for keywords like "tables", "columns", "schema", "structure", "what data do you have".
- 'OTHER': The query is COMPLETELY unrelated to the database, the domain, or general conversation about the assistant's capabilities.

Output ONLY one of the exact category names: 'DATA_QUERY', 'CONVERSATIONAL', 'SCHEMA_HELP', 'OTHER'.
Do NOT include any other text, explanations, or punctuation.
If the intent is ambiguous or doesn't fit clearly into the categories, default to 'OTHER'.
"""

        # --- Create messages list for the LLM including few-shot examples and history ---
        # Few-shot examples for classification
        few_shot_examples_classification = [
            {"role": "user", "content": "Show me the top 10 customers."},
            {"role": "assistant", "content": "DATA_QUERY"},

            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "CONVERSATIONAL"},

            {"role": "user", "content": "What tables do you have?"},
            {"role": "assistant", "content": "SCHEMA_HELP"},

            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "OTHER"},

            {"role": "user", "content": "Can you give me a list of investors located in USA."}, # Your test query
            {"role": "assistant", "content": "DATA_QUERY"}, # The expected output

             {"role": "user", "content": "can you show me 10 list of investors located in usa."}, # Another phrasing
            {"role": "assistant", "content": "DATA_QUERY"},

            {"role": "user", "content": "can you show me 10 list of investors located in usa with contact details."}, # Another phrasing
            {"role": "assistant", "content": "DATA_QUERY"},

            {"role": "user", "content": "List 10 people who work at Google."},
            {"role": "assistant", "content": "DATA_QUERY"},

            {"role": "user", "content": "what is the business development?"}, # User query from logs
            {"role": "assistant", "content": "OTHER"}, # Based on previous classification

            {"role": "user", "content": "i am a bike shop owner in middle east can you give me some data to increse my business"}, # User query from logs
            {"role": "assistant", "content": "DATA_QUERY"}, # <--- FIX: Classify this as DATA_QUERY!

            {"role": "user", "content": "can you help me with data extraction"},
            {"role": "assistant", "content": "OTHER"},

            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "OTHER"},

            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "CONVERSATIONAL"},

            {"role": "user", "content": "how are you ?"},
            {"role": "assistant", "content": "CONVERSATIONAL"},

             # Example with history - shows the LLM how to interpret context
            # Assuming previous turns were: [{"role": "user", "content": "Show me top 10 people"}, {"role": "assistant", "content": "... data ..."}]
            # And current message is: "Now show me the ones from California."
            # Represent this as: history + [{"role": "user", "content": "Now show me the ones from California."}]
            # The *assistant* response in the few-shot should be the *classification* based on the *combination*.
            # {"role": "user", "content": "Show me top 10 software engineers."}, {"role": "assistant", "content": "DATA_QUERY"}, {"role": "user", "content": "Now show me the ones from California."}, {"role": "assistant", "content": "DATA_QUERY"}, # Example turn with history
        ]

        # Combine few-shot examples, conversation history, then the current user message
        # The history is passed in here as a list of {"role": "user/assistant", "content": "..."} dicts
        messages = few_shot_examples_classification + conversation_history + [{"role": "user", "content": user_message}]


        try:
             raw_response = await self.openai_client.generate_text( # <--- Use await
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=20, # Enough for a single word category + a little buffer
                 temperature=0.0, # Make it deterministic
                 # Add stop sequences if needed, e.g., stop=["\n"]
             )

             if raw_response:
                 # Process the response to extract the category string
                 classified_intent = raw_response.strip().upper()
                 # Validate the response is one of the expected categories
                 valid_intents = ['DATA_QUERY', 'CONVERSATIONAL', 'SCHEMA_HELP', 'OTHER']
                 if classified_intent in valid_intents:
                      logger.debug(f"LLM classified intent as: {classified_intent}")
                      return classified_intent
                 else:
                      logger.warning(f"LLM returned invalid intent format: '{raw_response}'. Expected one of {valid_intents}. Falling back to 'OTHER'.")
                      return 'OTHER'
             else:
                  logger.warning("LLM returned empty response for intent classification. Falling back to 'OTHER'.")
                  return 'OTHER'

        except Exception as e:
             logger.error(f"Error during intent classification LLM call: {e}", exc_info=True)
             return 'OTHER'


    async def _generate_sql(self, user_message: str, schema_info: dict, conversation_history: List[Dict[str, str]]):
        """
        Uses the OpenAIClient to generate a SQL query based on the user message, schema, and history asynchronously.
        Includes conversation history in the prompt.
        Requires self.openai_client and schema_info to be available.
        """
        if self.openai_client is None:
             logger.error("Attempted SQL generation but OpenAI client is not available.")
             raise Exception("AI service is unavailable for SQL generation.") # Raise to be caught by process_message

        if not schema_info:
             logger.error("Attempted SQL generation but schema_info is empty.")
             raise Exception("Database schema information is missing.")


        # --- Implement Sophisticated Prompt Engineering for SQL Generation ---

        # Format schema for the LLM prompt
        formatted_schema = self._format_schema_for_llm(schema_info)

        # --- Define Few-Shot Examples for SQL Generation ---
        # Format them as {"role": "user": ..., "role": "assistant": ...} messages
        # Use the exact schema and column names from your CREATE TABLE statement.
        # Include examples using CTEs if needed.
        # (Few-shot examples remain the same, copied from previous steps)
        few_shot_examples = [
            # Example 1: Investors
            {"role": "user", "content": "can you give me 10 list of investors located in usa."},
            {"role": "assistant", "content": """WITH RankedInvestors AS (
    SELECT
        ProfileId,
        person_full_name,
        person_email,
        job_title,
        person_profile_headline,
        person_location_city,
        person_location_state,
        person_location_country,
        organization_name,
        organization_linkedin_url,
        ROW_NUMBER() OVER (
            PARTITION BY person_full_name
            ORDER BY ProfileId
        ) AS rn
    FROM
        dbo.ProfileData
    WHERE
        person_location_country = 'United States'
        AND (
            job_title LIKE '%investor%'
            OR person_profile_headline LIKE '%investor%'
            OR person_profile_headline LIKE '%venture%'
            OR person_industries LIKE '%investment%'
            OR person_industries LIKE '%venture%'
        )
)
SELECT TOP 10 *
FROM RankedInvestors
WHERE rn = 1;"""}, # Add a newline to separate few-shot examples

            # Example 2: Software Engineers
            {"role": "user", "content": "Give me 10 profiles of people working in software engineering."},
            {"role": "assistant", "content": """WITH RankedEngineers AS (
    SELECT
        ProfileId,
        person_full_name,
        person_email,
        job_title,
        person_profile_headline,
        person_location_city,
        person_location_state,
        person_location_country,
        organization_name,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        job_title IS NOT NULL AND (
            job_title LIKE '%software engineer%' OR
            job_title LIKE '%SDE%' OR
            job_title LIKE '%developer%' OR
            job_title LIKE '%programmer%'
        )
        AND job_title NOT LIKE '%CEO%'
        AND job_title NOT LIKE '%chief%'
        AND job_title NOT LIKE '%founder%'
        AND job_title NOT LIKE '%owner%'
)
SELECT TOP 10 *
FROM RankedEngineers
WHERE rn = 1;"""},

            # Example 3: Google Employees
             {"role": "user", "content": "List 10 people who work at Google."},
            {"role": "assistant", "content": """WITH RankedGoogleEmployees AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        organization_name,
        person_email,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE organization_name LIKE '%Google%'
)
SELECT TOP 10 *
FROM RankedGoogleEmployees
WHERE rn = 1;"""},

            # Example 4: CTOs in US
            {"role": "user", "content": "Show 10 CTOs from companies based in the United States."},
            {"role": "assistant", "content": """WITH RankedCTOs AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        organization_name,
        person_location_city,
        person_location_state,
        person_location_country,
        person_email,
        person_profile_headline,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        person_location_country = 'United States'
        AND (
            -- Primary CTO identifiers
            LOWER(job_title) = 'cto' OR
            LOWER(job_title) LIKE '%chief technology officer%' OR
            LOWER(job_title) LIKE '%chief tech officer%' OR
            LOWER(job_title) LIKE '%c.t.o%' OR

            -- Allow some expanded CTO-like roles
            LOWER(job_title) LIKE '%technical co-founder%' OR
            LOWER(job_title) LIKE '%technical founder%' OR
            LOWER(job_title) LIKE '%vp of engineering%' OR
            LOWER(job_title) LIKE '%head of engineering%' OR
            LOWER(job_title) LIKE '%head of technology%'
        )
        -- Exclude irrelevant director/VP types
        AND LOWER(job_title) NOT LIKE '%director of marketing%'
        AND LOWER(job_title) NOT LIKE '%marketing%'
        AND LOWER(job_title) NOT LIKE '%sales%'
        AND LOWER(job_title) NOT LIKE '%business development%'
        AND LOWER(job_title) NOT LIKE '%finance%'
        AND LOWER(job_title) NOT LIKE '%account%'
)
SELECT TOP 10 *
FROM RankedCTOs
WHERE rn = 1;"""},

            {"role": "user", "content": "Retrieve 10 startup founders from any country."},
            {"role": "assistant", "content": """WITH RankedFounders AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        person_profile_headline,
        organization_name,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        job_title LIKE '%founder%'
        OR person_profile_headline LIKE '%founder%'
)
SELECT TOP 10 *
FROM RankedFounders
WHERE rn = 1;"""},

            {"role": "user", "content": "Show 10 marketing professionals from Europe."},
            {"role": "assistant", "content": """WITH RankedMarketing AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        person_location_continent,
        organization_name,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        person_location_continent = 'Europe'
        AND (
            job_title LIKE '%marketing%' OR
            person_profile_headline LIKE '%marketing%'
        )
)
SELECT TOP 10 *
FROM RankedMarketing
WHERE rn = 1;"""},

            {"role": "user", "content": "Get 10 profiles where the person uses GitHub."},
            {"role": "assistant", "content": """WITH RankedGitHubUsers AS (
    SELECT
        ProfileId,
        person_full_name,
        person_github_url,
        person_github_username,
        job_title,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        person_github_url IS NOT NULL
        AND person_github_url <> ''
)
SELECT TOP 10 *
FROM RankedGitHubUsers
WHERE rn = 1;"""},

            {"role": "user", "content": "List 10 HR managers from organizations with more than 500 employees."},
            {"role": "assistant", "content": """WITH RankedHRManagers AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        organization_name,
        organization_size,
        person_location_country,
        person_email,
        person_profile_headline,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        organization_size IN (
            '501-1000',
            '1001-5000',
            '5001-10000',
            '10001+'
        )
        AND (
            LOWER(job_title) LIKE '%hr manager%' OR
            LOWER(job_title) LIKE '%human resources manager%' OR
            LOWER(job_title) LIKE '%people manager%' OR
            LOWER(job_title) LIKE '%talent manager%' OR
            LOWER(job_title) LIKE '%recruiting manager%'
        )
)
SELECT TOP 100 *
FROM RankedHRManagers
WHERE rn = 1;"""}, # Note: Example requested TOP 10 but provided TOP 100. LLM might learn 100 from this. Consistency helps.

            {"role": "user", "content": "Give me 10 profiles with active personal email addresses."},
            {"role": "assistant", "content": """WITH RankedEmailProfiles AS (
    SELECT
        ProfileId,
        person_full_name,
        person_email,
        person_email_status,
        job_title,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        person_email_status = 'valid'
)
SELECT TOP 10 *
FROM RankedEmailProfiles
WHERE rn = 1;"""},

            {"role": "user", "content": "Find 10 people who recently updated their job title."},
            {"role": "assistant", "content": """WITH RankedRecentUpdates AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        job_last_updated,
        organization_name,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        job_last_updated IS NOT NULL
        AND TRY_CAST(job_last_updated AS datetime) > DATEADD(MONTH, -6, GETDATE())
)
SELECT TOP 10 *
FROM RankedRecentUpdates
WHERE rn = 1;"""},

            {"role": "user", "content": "can you give me 10 ceo list located in USA with contact details"},
            {"role": "assistant", "content": """WITH RankedCEOs AS (
    SELECT
        ProfileId,
        person_full_name,
        job_title,
        person_email,
        person_phone,
        person_mobile,
        person_linkedin_url,
        organization_name,
        organization_linkedin_url,
        person_location_city,
        person_location_state,
        person_location_country,
        ROW_NUMBER() OVER (PARTITION BY person_full_name ORDER BY ProfileId) AS rn
    FROM dbo.ProfileData
    WHERE
        person_location_country = 'United States'
        AND (
            LOWER(job_title) LIKE 'ceo' OR
            LOWER(job_title) LIKE '%chief executive officer%' OR
            LOWER(job_title) LIKE '%founder and ceo%' OR
            LOWER(job_title) LIKE 'co-founder & ceo%' OR
            LOWER(job_title) LIKE '%co-founder and ceo%' OR
            LOWER(job_title) LIKE '%ceo & founder%' OR
            LOWER(job_title) LIKE '%ceo/co-founder%'
        )
)
SELECT TOP 10 *
FROM RankedCEOs
WHERE rn = 1;"""},
             # --- Add more examples based on your schema and user needs ---
             {"role": "user", "content": "Count the number of people with a valid personal email."},
             {"role": "assistant", "content": "SELECT COUNT(*) FROM dbo.ProfileData WHERE person_email_status = 'valid';"},

             {"role": "user", "content": "Show me the top 5 people with a LinkedIn URL and their job title."},
             {"role": "assistant", "content": "SELECT TOP 5 person_full_name, job_title, person_linkedin_url FROM dbo.ProfileData WHERE person_linkedin_url IS NOT NULL AND person_linkedin_url <> '';"},

             {"role": "user", "content": "List names and organizations for people in California."},
             {"role": "assistant", "content": "SELECT TOP 100 person_full_name, organization_name FROM dbo.ProfileData WHERE person_location_state = 'California';"}, # Assuming 'California' is stored in person_location_state

             {"role": "user", "content": "Show me the top 20 profiles in New York city."},
             {"role": "assistant", "content": "SELECT TOP 20 * FROM dbo.ProfileData WHERE person_location_city = 'New York';"},

             {"role": "user", "content": "Count profiles updated in 2024."},
             {"role": "assistant", "content": "SELECT COUNT(*) FROM dbo.ProfileData WHERE YEAR(updated_on) = 2024;"}, # Using YEAR function

             {"role": "user", "content": "List job titles and organizations for the first 50 records sorted by ProfileId."},
             {"role": "assistant", "content": "SELECT TOP 50 job_title, organization_name FROM dbo.ProfileData ORDER BY ProfileId;"},

             # --- Add examples for follow-up queries using history ---
             # Example 18 (Follow-up): After "Show me the top 10 software engineers." -> "Now show me the ones from California."
             # This is the key for history! Show the LLM how to combine current message + previous context.
             # The prompt messages will look like: [System], [Few-shots], [User: "Show me... engineers"], [Assistant: "SELECT..."], [User: "Now show me... California."]
             # The LLM needs to generate SQL for the *last* user message, using the preceding messages as context.
             # This example *shows* the history pattern and the expected SQL *given that history*.
             {"role": "user", "content": "Show me the top 10 software engineers."},
             {"role": "assistant", "content": "SELECT TOP 10 ProfileId, person_full_name, job_title FROM dbo.ProfileData WHERE job_title LIKE '%software engineer%' OR job_title LIKE '%SDE%';"}, # Simplified first turn SQL for example

             {"role": "user", "content": "Now show me the ones from California."},
             {"role": "assistant", "content": "SELECT TOP 10 ProfileId, person_full_name, job_title FROM dbo.ProfileData WHERE (job_title LIKE '%software engineer%' OR job_title LIKE '%SDE%') AND person_location_state = 'California';"}, # Expected combined SQL

             # Example 19 (Another Follow-up): After "List 10 people who work at Google." -> "How many of them have a valid email?"
             {"role": "user", "content": "List 10 people who work at Google."},
             {"role": "assistant", "content": "SELECT TOP 10 ProfileId, person_full_name FROM dbo.ProfileData WHERE organization_name LIKE '%Google%';"}, # Simplified first turn SQL for example

             {"role": "user", "content": "How many of them have a valid email?"},
             {"role": "assistant", "content": "SELECT COUNT(*) FROM dbo.ProfileData WHERE organization_name LIKE '%Google%' AND person_email_status = 'valid';"}, # Expected combined SQL

             # --- Add examples specifically targeting the "bike shop owner" scenario themes ---
             {"role": "user", "content": "i am a bike shop owner in middle east i wan groww my businees give me some data"},
             {"role": "assistant", "content": "SELECT TOP 100 person_full_name, job_title, person_email, person_phone, organization_name, organization_location_country, organization_industries FROM dbo.ProfileData WHERE (job_title LIKE '%sales%' OR job_title LIKE '%business development%' OR job_title LIKE '%marketing%') AND person_location_continent = 'Asia' AND organization_industries LIKE '%Retail%';"}, # Example mapping user context to relevant data

             {"role": "user", "content": "Find people in the retail industry in Dubai."}, # More specific version
             {"role": "assistant", "content": "SELECT TOP 100 person_full_name, job_title, organization_name, person_location_city, organization_industries FROM dbo.ProfileData WHERE organization_industries LIKE '%Retail%' AND person_location_city = 'Dubai';"},

             {"role": "user", "content": "List contact details for sales people in Europe."}, # Another variation
             {"role": "assistant", "content": "SELECT TOP 100 person_full_name, person_email, person_phone, person_linkedin_url, job_title FROM dbo.ProfileData WHERE (job_title LIKE '%sales%' OR job_title LIKE '%business development%') AND person_location_continent = 'Europe';"},


        ]
        # --- End Few-Shot Examples ---


        # Example prompt structure:
        system_instructions = f"""You are a helpful assistant that writes T-SQL queries for a SQL Server database.
Your task is to convert natural language questions into T-SQL queries based on the provided schema, conversation history, and examples.

The database schema is:
{formatted_schema}

IMPORTANT INSTRUCTIONS:
- **Consider the conversation history provided in the messages.** If the current user message is a follow-up question (e.g., "Now show me the ones from California"), combine the context from previous turns (user and assistant messages) with the current message to generate the appropriate SQL query.
- **Map user roles, goals, or industries to relevant data points.** For example, "bike shop owner looking to grow business" might imply searching for contacts in the 'Retail' industry, or in roles like 'sales', 'business development', or 'marketing'.
- Only generate valid T-SQL SELECT statements.
- Do NOT generate any other types of statements (INSERT, UPDATE, DELETE, DROP, ALTER, EXEC, etc.).
- Always include a TOP clause to limit the number of results (e.g., `SELECT TOP 100 ...`). Use 100 as the default TOP value unless the user specifies a different limit (e.g., "top 5", "first 20").
- Do NOT include any comments (-- or /* */) in the generated SQL.
- Do NOT generate multiple statements separated by semicolons (;).
- Ensure table and column names match the schema exactly (case-insensitive names are fine in T-SQL queries). If a column name has spaces or special characters, enclose it in square brackets `[]`.
- Use appropriate JOINs if the query requires data from multiple tables (based on relationships you might describe or imply through schema - *currently only one table loaded*).
- If the user asks for a count, use `COUNT(*)`.
- If the user asks for filtering by date, use standard T-SQL date functions or literal date strings (e.g., `created_on > '2023-01-01'`). Use functions like `YEAR()`, `MONTH()`, `GETDATE()`, `DATEADD()`, `TRY_CAST()` as needed for date comparisons based on the examples.
- If you cannot generate a safe and relevant SELECT query based on the schema, history, and the user's request, respond with ONLY the specific message: "I cannot generate a query for that based on the available data."

Follow the provided examples to understand the desired output format and query style, including how to handle context from history and map user intent to query filters/columns.
"""

        # Create messages list for the LLM
        # Add few-shot examples, conversation history, then the current user message
        messages = few_shot_examples + conversation_history + [{"role": "user", "content": user_message}]

        # Check the final message structure being sent to LLM (for debugging)
        logger.debug(f"Messages sent to LLM for SQL generation: {messages}")


        try:
             raw_response = await self.openai_client.generate_text( # <--- Use await
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=1000, # Increased max_tokens
                 temperature=0.05, # Low temperature for SQL generation
                 # Add stop sequences if needed
             )

             if raw_response:
                 generated_sql = raw_response.strip()
                 if generated_sql.startswith("```sql"):
                      generated_sql = generated_sql.replace("```sql", "").strip()
                 if generated_sql.endswith("```"):
                      generated_sql = generated_sql.replace("```", "").strip()
                 if generated_sql.endswith(";"):
                     generated_sql = generated_sql.rstrip(";")
                     logger.debug("Removed trailing semicolon.")

                 logger.debug(f"Raw LLM response for SQL generation: {raw_response}")
                 logger.debug(f"Cleaned generated SQL: {generated_sql}")

                 if "I CANNOT GENERATE A QUERY FOR THAT" in generated_sql.upper():
                     logger.warning(f"LLM indicated it cannot generate a query: {generated_sql}")
                     raise ValueError("LLM indicated it cannot generate a suitable query.")

                 return generated_sql

             else:
                  logger.error("LLM returned empty response for SQL generation.")
                  raise Exception("AI did not generate a valid SQL query.")

        except Exception as e:
             logger.error(f"Error during SQL generation LLM call: {e}", exc_info=True)
             if isinstance(e, ValueError):
                 raise e
             else:
                 raise Exception(f"Failed to generate SQL query: {e}") from e

    # --- Implement async def _generate_conversational_response method ---
    async def _generate_conversational_response(self, user_message: str, conversation_history: List[Dict[str, str]]):
        # ... (This method remains the same) ...
        if self.openai_client is None:
             logger.error("Attempted conversational generation but OpenAI client is not available.")
             raise Exception("AI service is unavailable for conversation.")

        system_instructions = "You are a friendly and helpful AI assistant for a company that manages profile data. You can answer general questions but cannot perform actions or access external websites. Keep responses concise."

        messages = conversation_history + [{"role": "user", "content": user_message}]

        try:
             raw_response = await self.openai_client.generate_text(
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=200,
                 temperature=0.7,
             )

             if raw_response:
                 return raw_response.strip()
             else:
                  logger.warning("LLM returned empty response for conversational generation.")
                  return "I'm not sure how to respond to that."

        except Exception as e:
             logger.error(f"Error during conversational LLM call: {e}", exc_info=True)
             raise Exception(f"Failed to generate conversational response: {e}") from e

    # --- Implement schema formatting methods ---
    def _format_schema_for_llm(self, schema_info: dict) -> str:
         # ... (This method remains the same) ...
        formatted_string = ""
        if not schema_info:
            return "No database schema available."

        formatted_string += "Available Tables and Columns:\n"
        for table_name, columns in schema_info.items():
             formatted_string += f"- Table: [{table_name}]\n"
             if columns:
                  formatted_string += "  Columns:\n"
                  for column in columns:
                       column_parts = column.split(' ')
                       column_name = column_parts[0]
                       data_type = ' '.join(column_parts[1:])
                       formatted_string += f"  - [{column_name}] {data_type}\n"
             else:
                  formatted_string += "  (No columns found)\n"

        logger.debug("Formatted schema for LLM.")
        return formatted_string

    def _format_schema_for_user(self, schema_info: dict) -> str:
        # ... (This method remains the same) ...
        if not schema_info:
            return "I was able to access the database, but it appears to be empty or I couldn't load the schema."

        formatted_string = "Here are the tables and columns I can access:\n\n"
        for table_name, columns in schema_info.items():
             formatted_string += f"**{table_name}**\n"
             if columns:
                 for column in columns:
                      column_parts = column.split(' ')
                      column_name = column_parts[0]
                      data_type = ' '.join(column_parts[1:])
                      formatted_string += f"- `{column_name}` {data_type}\n"
             else:
                  formatted_string += "- (No columns)\n"
             formatted_string += "\n"
        logger.debug("Formatted schema for user.")
        return formatted_string

# Note: No __main__ block in this file