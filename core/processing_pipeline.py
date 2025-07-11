# core/processing_pipeline.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
print("DEBUG: Loading core/processing_pipeline.py - Version Check 7") # Increment version
# --- END ADDITION ---

import logging
import inspect
import pandas as pd

from utils.config_loader import AppConfig
from integrations.database.db_connector import DatabaseConnector
from integrations.database.schema_loader import SchemaLoader
from integrations.openai.openai_client import OpenAIClient
from integrations.database.sql_safety import SQLSafetyChecker
from integrations.database.query_executor import QueryExecutor

from builtins import RuntimeError

logger = logging.getLogger(__name__)

class ConversationPipeline:
    def __init__(
        self,
        app_config: AppConfig,
        db_connector: DatabaseConnector,
        schema_loader: SchemaLoader,
        openai_client: OpenAIClient | None,
        sql_safety_checker: SQLSafetyChecker | None,
    ):
        if not isinstance(app_config, AppConfig):
             raise TypeError("app_config must be an instance of AppConfig")
        if not isinstance(db_connector, DatabaseConnector):
             raise TypeError("db_connector must be an instance of DatabaseConnector")
        if not isinstance(schema_loader, SchemaLoader):
             raise TypeError("schema_loader must be an instance of SchemaLoader")

        self.app_config = app_config
        self.db_connector = db_connector
        self.schema_loader = schema_loader
        self.openai_client = openai_client
        self.sql_safety_checker = sql_safety_checker

        if self.db_connector:
             self.query_executor = QueryExecutor(self.db_connector)
        else:
             self.query_executor = None
             logger.warning("QueryExecutor not initialized due to missing DatabaseConnector.")

        logger.info("ConversationPipeline initialized.")

    async def process_message(self, user_message: str, conversation_id: str | None = None):
        logger.info(f"Processing message: '{user_message}' for conversation {conversation_id}")

        if self.openai_client is None:
             logger.error("OpenAI client is not available. Cannot classify intent or generate responses.")
             return {'type': 'text', 'content': "Sorry, the AI service is currently unavailable."}

        intent = await self._classify_intent(user_message, conversation_id)
        logger.info(f"Message classified with intent: {intent}")

        if intent == 'DATA_QUERY':
            logger.info("Intent is DATA_QUERY. Proceeding with SQL generation.")

            if self.db_connector is None or self.schema_loader is None or self.query_executor is None or self.sql_safety_checker is None:
                 logger.error("Critical services for data query are unavailable (DB, Schema, Executor, Safety Checker).")
                 return {'type': 'text', 'content': "Sorry, I'm unable to access the database to answer your question right now."}

            try:
                schema_info = self.schema_loader.get_full_schema()
                logger.debug(f"Loaded schema for {len(schema_info)} tables.")
                if not schema_info:
                     logger.warning("Loaded schema is empty. Cannot generate meaningful queries.")
                     return {'type': 'text', 'content': "Sorry, I was able to connect to the database, but it appears to be empty or I couldn't load the schema correctly."}

            except Exception as e:
                 logger.error(f"Failed to load database schema during query processing: {e}", exc_info=True)
                 return {'type': 'text', 'content': "Sorry, I'm having trouble accessing the database schema right now. Please try again later."}

            try:
                generated_sql = await self._generate_sql(user_message, schema_info, conversation_id)
                logger.info(f"Generated SQL: {generated_sql}")
            except Exception as e:
                 logger.error(f"Failed to generate SQL using OpenAI: {e}", exc_info=True)
                 return {'type': 'text', 'content': "Sorry, I'm having trouble generating a database query right now. Please try again later."}

            is_safe = self.sql_safety_checker.is_safe(generated_sql, schema_info)

            if not is_safe:
                logger.warning(f"Generated SQL deemed unsafe: {generated_sql}")
                raise ValueError("The generated query was deemed unsafe.")

            try:
                results = self.query_executor.execute_select(generated_sql)

                logger.info(f"Query executed successfully. Results: {type(results)}")

                return {'type': 'data', 'content': results, 'generated_sql': generated_sql}

            except RuntimeError as e:
                logger.error(f"Database query execution failed in pipeline: {e}")
                raise e

        elif intent == 'CONVERSATIONAL':
            logger.info("Intent is CONVERSATIONAL. Proceeding with text response.")
            try:
                conversational_response = await self._generate_conversational_response(user_message, conversation_id)
                logger.debug(f"Generated conversational response: {conversational_response}")
                return {'type': 'text', 'content': conversational_response}
            except Exception as e:
                logger.error(f"Failed to generate conversational response using OpenAI: {e}", exc_info=True)
                return {'type': 'text', 'content': "Sorry, I'm having trouble responding right now."}

        elif intent == 'SCHEMA_HELP':
             logger.info("Intent is SCHEMA_HELP.")
             if self.schema_loader is None:
                 logger.error("SchemaLoader is not available. Cannot provide schema help.")
                 return {'type': 'text', 'content': "Sorry, I'm unable to provide details about the database schema because the schema loader is not available."}

             try:
                 schema_info = self.schema_loader.get_full_schema()
                 if not schema_info:
                      logger.warning("Schema help requested, but loaded schema is empty.")
                      return {'type': 'text', 'content': "I was able to connect to the database, but it appears to be empty or I couldn't load the schema correctly."}

                 schema_text = self._format_schema_for_user(schema_info)
                 logger.debug("Formatted schema for user.")
                 return {'type': 'text', 'content': schema_text}
             except Exception as e:
                  logger.error(f"Failed to provide schema help: {e}", exc_info=True)
                  return {'type': 'text', 'content': "Sorry, an unexpected error occurred while trying to get schema details."}

        elif intent == 'OTHER' or intent == 'UNCLEAR':
            logger.info("Intent is OTHER/UNCLEAR.")
            return {'type': 'text', 'content': "I can help you with questions about our data. What information are you looking for?"}

        else:
            logger.error(f"Unknown intent returned by classifier: {intent}")
            return {'type': 'text', 'content': "Sorry, I didn't understand that."}

    # --- Private Helper Methods (Implementing with LLM calls) ---

    async def _classify_intent(self, user_message: str, conversation_id: str | None):
        """
        Uses the OpenAIClient to classify the user's intent asynchronously.
        Possible return values: 'DATA_QUERY', 'CONVERSATIONAL', 'SCHEMA_HELP', 'OTHER', 'UNCLEAR'.
        Requires self.openai_client to be available.
        """
        logger.debug(f"Classifying intent for message: '{user_message}'")

        if self.openai_client is None:
             logger.error("Attempted intent classification but OpenAI client is not available.")
             return 'OTHER'

        # --- Refined system_instructions for classification ---
        system_instructions = """You are an intent classifier for a database AI assistant.
Your task is to analyze user queries and categorize them into one of the following intents:

- 'DATA_QUERY': The user is asking for specific data FROM the database.
- 'CONVERSATIONAL': The user is engaging in general chat, greetings, or non-data related questions ABOUT the domain or the assistant itself.
- 'SCHEMA_HELP': The user is asking about the STRUCTURE of the database (tables, columns, data types, what data is available).
- 'OTHER': The query is COMPLETELY unrelated to the database, the domain, or general conversation about the assistant's capabilities.

Output ONLY one of the exact category names: 'DATA_QUERY', 'CONVERSATIONAL', 'SCHEMA_HELP', 'OTHER'.
Do NOT include any other text, explanations, or punctuation.
If the intent is ambiguous or doesn't fit clearly into the categories, default to 'OTHER'.
"""

        # --- Add More Few-Shot Examples for Classification ---
        # Show the LLM exactly what output format is expected for different inputs
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
            {"role": "assistant", "content": "OTHER"}, # Based on previous classification (too vague)

            {"role": "user", "content": "can you help me with data extraction"}, # User query from logs
            {"role": "assistant", "content": "OTHER"}, # Based on previous classification (too vague)

            {"role": "user", "content": "What is the capital of France?"}, # Unrelated query
            {"role": "assistant", "content": "OTHER"},

            # Add more examples covering different nuances for your domain
        ]
        # --- End Few-Shot Examples for Classification ---


        # Combine few-shot examples with the current user message
        messages = few_shot_examples_classification + [{"role": "user", "content": user_message}]

        # TODO: Include conversation history if available for better context
        # if conversation_history:
        #    messages = conversation_history + messages # Add history after examples


        try:
             raw_response = await self.openai_client.generate_text(
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=20,
                 temperature=0.0, # Make it deterministic
             )

             if raw_response:
                 classified_intent = raw_response.strip().upper()
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


    async def _generate_sql(self, user_message: str, schema_info: dict, conversation_id: str | None):
        """
        Uses the OpenAIClient to generate a SQL query based on the user message and schema asynchronously.
        Requires self.openai_client and schema_info to be available.
        """
        if self.openai_client is None:
             logger.error("Attempted SQL generation but OpenAI client is not available.")
             raise Exception("AI service is unavailable for SQL generation.")

        if not schema_info:
             logger.error("Attempted SQL generation but schema_info is empty.")
             raise Exception("Database schema information is missing.")


        # --- Implement Sophisticated Prompt Engineering for SQL Generation ---

        # Format schema for the LLM prompt
        formatted_schema = self._format_schema_for_llm(schema_info)

        # --- Define Few-Shot Examples using the user-provided queries ---
        # Format them as {"role": "user": ..., "role": "assistant": ...} messages
        # Use the exact schema and column names from your CREATE TABLE statement.
        few_shot_examples = [
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
        ]
        # --- End Few-Shot Examples ---


        # Example prompt structure:
        # Refine this prompt significantly!
        system_instructions = f"""You are a helpful assistant that writes T-SQL queries for a SQL Server database.
Your task is to convert natural language questions into T-SQL queries based on the provided schema and examples.

The database schema is:
{formatted_schema}

IMPORTANT INSTRUCTIONS:
- Only generate valid T-SQL SELECT statements.
- Do NOT generate any other types of statements (INSERT, UPDATE, DELETE, DROP, ALTER, EXEC, etc.).
- Always include a TOP clause to limit the number of results (e.g., `SELECT TOP 100 ...`). Use 100 as the default TOP value unless the user specifies a different limit (e.g., "top 5", "first 20").
- Do NOT include any comments (-- or /* */) in the generated SQL.
- Do NOT generate multiple statements separated by semicolons (;).
- Ensure table and column names match the schema exactly (case-insensitive names are fine in T-SQL queries). If a column name has spaces or special characters, enclose it in square brackets `[]`.
- Use appropriate JOINs if the query requires data from multiple tables (based on relationships you might describe or imply through schema - *currently only one table loaded*).
- If the user asks for a count, use `COUNT(*)`.
- If the user asks for filtering by date, use standard T-SQL date functions or literal date strings (e.g., `created_on > '2023-01-01'`).
- If you cannot generate a safe and relevant SELECT query based on the schema and the user's request, respond with ONLY the specific message: "I cannot generate a query for that based on the available data."

Follow the provided examples to understand the desired output format and query style.
"""

        # Create messages list for the LLM
        # Add the few-shot examples first, then the current user message
        messages = few_shot_examples + [{"role": "user", "content": user_message}]

        # TODO: Include conversation history if available (requires conversation_manager)
        # if conversation_history:
        #    messages = conversation_history + messages # Add history *after* examples

        try:
             # Use a lower temperature for more deterministic (SQL) output
             raw_response = await self.openai_client.generate_text( # <--- Use await
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=1000, # Increased max_tokens to accommodate longer CTE queries
                 temperature=0.05, # Low temperature for SQL generation
                 # Add stop sequences if needed, e.g., stop=[";", "\n\n"]
             )

             if raw_response:
                 # Basic post-processing (remove leading/trailing whitespace, markdown code block markers)
                 generated_sql = raw_response.strip()
                 # Remove markdown code block if present
                 if generated_sql.startswith("```sql"):
                      generated_sql = generated_sql.replace("```sql", "").strip()
                 if generated_sql.endswith("```"):
                      generated_sql = generated_sql.replace("```", "").strip()
                 # Remove trailing semicolon if present (although prompt asks not to include it)
                 if generated_sql.endswith(";"):
                     generated_sql = generated_sql.rstrip(";")
                     logger.debug("Removed trailing semicolon.")


                 logger.debug(f"Raw LLM response for SQL generation: {raw_response}")
                 logger.debug(f"Cleaned generated SQL: {generated_sql}")

                 # Check if the LLM returned the "cannot generate" message (case-insensitive check)
                 if "I CANNOT GENERATE A QUERY FOR THAT" in generated_sql.upper():
                     logger.warning(f"LLM indicated it cannot generate a query: {generated_sql}")
                     # Raise the ValueError as planned
                     raise ValueError("LLM indicated it cannot generate a suitable query.")


                 # Return the generated SQL string. This will be passed to the safety checker.
                 return generated_sql

             else:
                  logger.error("LLM returned empty response for SQL generation.")
                  raise Exception("AI did not generate a valid SQL query.") # Raise to be caught by process_message

        except Exception as e:
             logger.error(f"Error during SQL generation LLM call: {e}", exc_info=True)
             # Re-raise specific ValueErrors caught above, otherwise wrap in Exception
             if isinstance(e, ValueError): # If it's our ValueError from LLM fallback message
                 raise e # Re-raise the ValueError directly
             else:
                 raise Exception(f"Failed to generate SQL query: {e}") # Re-raise other exceptions

    # --- Implement async def _generate_conversational_response method ---
    async def _generate_conversational_response(self, user_message: str, conversation_id: str | None):
        """
        Uses the OpenAIClient to generate a general conversational text response asynchronously.
        Requires self.openai_client to be available.
        """
        if self.openai_client is None:
             logger.error("Attempted conversational generation but OpenAI client is not available.")
             raise Exception("AI service is unavailable for conversation.") # Raise to be caught by process_message

        # TODO: Implement Sophisticated Conversational Prompt Engineering using LLM call
        # Craft a prompt for general conversation, potentially including domain context.
        # TODO: Include conversation history
        system_instructions = "You are a friendly and helpful AI assistant for a company that manages profile data. You can answer general questions but cannot perform actions or access external websites. Keep responses concise."
        messages = [{"role": "user", "content": user_message}]
        # if conversation_history:
        #    messages = conversation_history + messages # Add history before the current message


        try:
             # Use a higher temperature for more creative/human-like responses
             raw_response = await self.openai_client.generate_text( # <--- Use await
                 messages,
                 custom_instructions=system_instructions,
                 max_tokens=200, # Adjust max_tokens for expected response length
                 temperature=0.7, # Higher temperature for chat
             )

             if raw_response:
                 return raw_response.strip()
             else:
                  logger.warning("LLM returned empty response for conversational generation.")
                  return "I'm not sure how to respond to that." # Fallback response

        except Exception as e:
             logger.error(f"Error during conversational LLM call: {e}", exc_info=True)
             raise Exception(f"Failed to generate conversational response: {e}") # Re-raise


    # --- Implement schema formatting methods (basic placeholders provided) ---
    def _format_schema_for_llm(self, schema_info: dict) -> str:
         """Formats schema dictionary into a string suitable for including in LLM prompt."""
         formatted_string = "" # Start empty, system instructions add context
         if not schema_info:
             return "No database schema available." # Handle empty schema case

         # Use a format that is easy for the LLM to parse and understand as T-SQL structure
         formatted_string += "Available Tables and Columns:\n"
         # Iterate through tables and columns
         for table_name, columns in schema_info.items():
              # Use T-SQL syntax for table names (often includes schema like dbo.TableName)
              # Use brackets for safety, even if not needed by LLM
              formatted_string += f"- Table: [{table_name}]\n"
              # List columns with data types
              if columns:
                   formatted_string += "  Columns:\n"
                   for column in columns:
                        # Format: - [ColumnName] (DataType)
                        column_parts = column.split(' ')
                        column_name = column_parts[0]
                        data_type = ' '.join(column_parts[1:])
                        # Use brackets for column names
                        formatted_string += f"  - [{column_name}] {data_type}\n"
              else:
                   formatted_string += "  (No columns found)\n" # Handle tables with no columns

         # TODO: Add relationships if you can load them from the database
         # Example: formatted_string += "\nTable Relationships (Primary Key -> Foreign Key):\n"
         # Example: formatted_string += "- [dbo.Customers].[CustomerId] -> [dbo.Orders].[CustomerId]\n"

         logger.debug("Formatted schema for LLM.")
         return formatted_string

    def _format_schema_for_user(self, schema_info: dict) -> str:
        """Formats schema dictionary into a user-friendly string response."""
        if not schema_info:
            return "I was able to access the database, but it appears to be empty or I couldn't load the schema."

        formatted_string = "Here are the tables and columns I can access:\n\n"
        for table_name, columns in schema_info.items():
             formatted_string += f"**{table_name}**\n" # Use Markdown for emphasis
             if columns:
                 for column in columns:
                      # Display column name and type
                      column_parts = column.split(' ')
                      column_name = column_parts[0]
                      data_type = ' '.join(column_parts[1:])
                      formatted_string += f"- `{column_name}` {data_type}\n" # Format: `column_name` (data type)
             else:
                  formatted_string += "- (No columns)\n"
             formatted_string += "\n"
        logger.debug("Formatted schema for user.")
        return formatted_string


# Note: This file does not have a __main__ block to run tests directly
# because the pipeline requires initialized service instances provided by main.py.
# Testing the pipeline logic is typically done via integration tests
# that simulate the FastAPI endpoint call or instantiate the pipeline with mock services.