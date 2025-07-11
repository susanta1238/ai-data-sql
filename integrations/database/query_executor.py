# integrations/database/query_executor.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
# Add this line to help debug if this specific file version is being loaded.
# If this message doesn't appear in your terminal logs during startup or test,
# it means an older version of this file is being used due to caching or other environmental issue.
print("DEBUG: Loading integrations/database/query_executor.py - Version Check 16") # Increment version
# --- END ADDITION ---


import logging
import pyodbc
import pandas as pd # Recommend using pandas for data handling
import os # Import os for test block

# Import DatabaseConnector from the same directory
from .db_connector import DatabaseConnector # Using relative import within the package

# You might also need the SQLSafetyChecker if you want the executor
# to perform a final check here, although the pipeline should handle
# the check *before* calling the executor.
# from .sql_safety import SQLSafetyChecker

logger = logging.getLogger(__name__) # Get logger at module level

class QueryExecutor:
    """
    Executes validated SQL queries against the database and returns results.
    Assumes queries passed through SQLSafetyChecker first.
    """
    def __init__(self, db_connector: DatabaseConnector):
        """
        Initializes the executor with a DatabaseConnector instance.
        """
        # Basic type check for db_connector instance
        if not isinstance(db_connector, DatabaseConnector):
             raise TypeError("db_connector must be an instance of DatabaseConnector")

        self.db_connector = db_connector
        logger.info("QueryExecutor initialized.")

    def execute_select(self, safe_sql_query: str) -> pd.DataFrame | str:
        """
        Executes a *validated* SELECT query against the database.

        Args:
            safe_sql_query (str): The SQL query string. Must be a safe SELECT query.
                                  Results should ideally be limited (TOP/LIMIT).

        Returns:
            pd.DataFrame: A pandas DataFrame containing the query results.
                          Returns an empty DataFrame if no results.
            str: A string message if the query executes successfully but returns no data.

        Raises:
            ValueError: If the query string is empty or contains only whitespace.
            RuntimeError: If there is a database execution error (e.g., SQL syntax error caught by DB, missing table/column).
            Exception: For any unexpected errors.
        """
        # --- FIX: Remove the overly strict startswith("SELECT") check ---
        # The SQLSafetyChecker is responsible for validating the statement type.
        # Just ensure the query string is not empty after stripping.
        if not safe_sql_query or not safe_sql_query.strip():
             logger.warning("QueryExecutor received empty query string.")
             return "The query string was empty." # Return message or raise ValueError? Return message is more user-friendly.


        conn = None
        cursor = None
        try:
            # Get a new connection for this query execution
            conn = self.db_connector.get_connection()
            cursor = conn.cursor()
            logger.debug(f"Executing query: {safe_sql_query}")

            # Execute the query
            # This is a SYNC call, happens blocking within the async FastAPI endpoint
            # If your DB driver/needs require true async, consider aioodbc or async SQLAlchemy
            cursor.execute(safe_sql_query)

            # Fetch results
            # Use cursor.fetchall() for all results or cursor.fetchmany(size)
            # For potentially large results, fetchmany and process in batches or stream
            # For simplicity and integration with pandas:
            rows = cursor.fetchall()

            # Get column names from the cursor description
            # Handle case where cursor.description is None (e.g., some non-SELECT statements, although safety checker should prevent this)
            if cursor.description is None:
                 # This case should ideally not happen with valid SELECT, but as a safeguard:
                 logger.warning("Query execution completed, but cursor.description is None. No results/columns found.")
                 # If no description and no rows, definitely no data
                 if not rows:
                     return "The query returned no data."
                 else:
                     # If rows were returned but no description, something is unexpected.
                     logger.error("Query returned rows but could not retrieve column information.")
                     # Attempt to return as records without column names, or raise error?
                     # Raising is safer:
                     raise RuntimeError("Query returned rows but could not retrieve column information.")


            # Extract column names
            columns = [column[0] for column in cursor.description]

            # Check if any rows were returned *after* processing description
            if not rows:
                logger.info("Query executed successfully but returned no data.")
                # Return a descriptive string for no data
                return "The query returned no data."

            # Create a pandas DataFrame from the results
            df = pd.DataFrame.from_records(rows, columns=columns)
            logger.info(f"Query executed successfully. Fetched {len(df)} rows.")

            # Note: If you need to enforce a MAX_ROWS_FETCH *after* fetching,
            # you would truncate the DataFrame here:
            # (Assuming max_result_rows is available via self.db_connector.config or directly)
            # max_display_rows = self.db_connector.config.get('max_result_rows', 1000)
            # if len(df) > max_display_rows:
            #      logger.warning(f"Result DataFrame size ({len(df)}) exceeds display/processing limit ({max_display_rows}). Truncating.")
            #      df = df.head(max_display_rows)


            # Return the DataFrame
            return df

        except pyodbc.ProgrammingError as ex:
            # pyodbc.ProgrammingError often indicates SQL syntax errors, missing tables/columns, etc.
            sqlstate = ex.args[0]
            logger.error(f"Database query execution failed (Programming Error). SQLSTATE: {sqlstate}", exc_info=True)
            # Raise a RuntimeError with a user-friendly message from the database error
            # The API endpoint will catch this and return a 500 (or 400 if appropriate)
            raise RuntimeError(f"Database query error: {ex.args[1]}") from ex
        except pyodbc.Error as ex:
            # Catch other pyodbc errors (connection issues *during* query, etc.)
            sqlstate = ex.args[0]
            logger.error(f"Database query execution failed (pyodbc Error). SQLSTATE: {sqlstate}", exc_info=True)
            raise RuntimeError(f"Database error during query execution: {ex.args[1]}") from ex
        except Exception as e:
            # Catch any other unexpected errors during execution or fetching
            logger.error(f"An unexpected error occurred during query execution: {e}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred during query execution: {e}") from e

        finally:
            # Ensure cursor and connection are closed in the finally block
            # Connection object is obtained from db_connector, so close it via db_connector
            if cursor:
                cursor.close()
            if conn:
                self.db_connector.close_connection(conn) # Use db_connector's method to close


# Example usage (Test Block) - Run with `python -m integrations.database.query_executor`
if __name__ == "__main__":
    # --- Imports needed specifically for this test block ---
    # These imports happen when the file is run directly via `python -m`
    import sys
    import os # Need os for DummyConfigForLoggingTest to read environment variables

    try:
        # Import utilities
        from utils.logging_config import setup_logging
        from utils.config_loader import load_config
        # Import required class from its absolute path within the package structure
        from integrations.database.db_connector import DatabaseConnector
        # Import the built-in ConnectionError exception for catching setup errors
        from builtins import ConnectionError

        logger.debug("Successfully imported utilities for test block.")

    except ImportError as e:
        # If core imports fail, print a fatal error and exit immediately
        print(f"FATAL ERROR: Failed to import required modules for test: {e}", file=sys.stderr)
        sys.exit(1)
    # --- End Imports for test block ---


    # --- Temporary Logging Setup for Test ---
    # This ensures logging works even when running this file directly via -m
    # It configures the module-level 'logger' obtained at the top of the file.
    try:
        # Need a dummy config object for logging setup
        # This dummy uses os.environ.get, so os MUST be imported above
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' }
                 # Add dummy sections that setup_logging might expect, even if empty
                 self.app = {}
                 self.openai = {}
                 # Include a dummy database section structure if needed by setup_logging or AppConfig init
                 self.database = {
                     'server': os.environ.get('DB_SERVER', 'YOUR_TEST_SERVER'),
                     'name': os.environ.get('DB_NAME', 'YOUR_TEST_DB'),
                     'trusted_connection': os.environ.get('DB_TRUSTED_CONNECTION', 'yes').lower() in ('true', '1', 'yes'),
                     'trust_server_certificate': os.environ.get('DB_TRUST_CERTIFICATE', 'yes').lower() in ('true', '1', 'yes'),
                     'odbc_driver': os.environ.get('DB_ODBC_DRIVER', '{ODBC Driver 18 for SQL Server}')
                 }

        # Only setup logging if it hasn't been configured globally yet
        # When running this file directly via -m, it's likely not configured,
        # so this block will run and configure the module-level logger.
        if not logging.getLogger().handlers:
            setup_logging(DummyConfigForLoggingTest())

        # Now that logging *is* configured, we can safely use the module-level 'logger' instance.
        logger.info("Temporary logging setup complete for query_executor test function.")

    except Exception as e:
        # Fallback if setting up temporary logging fails. Basic print is needed.
        print(f"Warning: Could not set up temporary logging in test function: {e}", file=sys.stderr)
        # Ensure a basic handler exists on the root logger if the temp setup failed too
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG) # Configure root logger
        # The module-level 'logger' instance still exists and will use the root logger's handlers
        logger.info("Using potentially basic logging for query_executor test function.")
    # --- End Temporary Logging Setup ---


    # --- Main Test Execution Block ---
    try:
        logger.info("--- Testing QueryExecutor ---")

        # Load configuration (Uses real config loading now - from .env/config.yaml)
        # This is where the actual database connection string is built based on your .env
        # This config object is also used to initialize DatabaseConnector
        app_config = load_config()
        logger.debug("Application configuration loaded for test.")

        # Initialize DatabaseConnector (QueryExecutor needs an instance)
        # This uses the database config loaded by load_config
        db_connector = DatabaseConnector(app_config)
        logger.debug("DatabaseConnector initialized for test.")

        # Initialize the QueryExecutor
        query_executor = QueryExecutor(db_connector)
        logger.debug("QueryExecutor initialized for test.")

        # Define test queries (ASSUME THESE ARE ALREADY VALIDATED AS SAFE SELECTS)
        # Replace with queries that work on *your* database's ProfileData table
        # Ensure dbo.ProfileData is accessible and contains data for SELECT tests
        test_queries = [
            "SELECT TOP 5 ProfileId, person_full_name, job_title FROM dbo.ProfileData;", # Valid SELECT, gets top 5 rows
            # Include a test query with a CTE that you provided, formatted exactly
            """WITH RankedInvestors AS (
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
WHERE rn = 1;""", # Valid SELECT with CTE

            "SELECT COUNT(*) FROM dbo.ProfileData WHERE organization_size = 'Large';", # Valid aggregate query
            "SELECT ProfileId, created_on FROM dbo.ProfileData WHERE created_on > '2023-01-01';", # Valid SELECT with date filter (string format)
            "SELECT ProfileId FROM dbo.ProfileData WHERE 1=0;", # Valid SELECT, but returns no rows
            # Uncomment to test a query that should cause a database error (e.g., table doesn't exist)
            # "SELECT * FROM NonExistentTable;",
        ]

        # Run tests
        for i, query in enumerate(test_queries):
            logger.info(f"\n--- Test Query {i+1}: '{query[:100]}...' ---") # Log snippet of query
            try:
                # Execute the query (assuming it's already validated as safe SELECT)
                # max_rows parameter is for DataFrame truncation *after* fetching, not SQL limit
                results = query_executor.execute_select(query) # , max_rows=100


                # Check results
                if isinstance(results, pd.DataFrame):
                    logger.info(f"Query returned a DataFrame with {len(results)} rows and {len(results.columns)} columns.")
                    if not results.empty:
                         logger.info("First 5 rows of results:")
                         # Print first few rows of the DataFrame using to_string for better console output
                         print(results.head().to_string())
                         logger.info("-" * 20) # Separator
                    else:
                         logger.info("DataFrame is empty.")
                elif isinstance(results, str):
                    logger.info(f"Query returned a message: {results}")
                else:
                    logger.warning(f"Query returned unexpected result type: {type(results)}")


            except RuntimeError as e: # Catch the custom RuntimeError raised by execute_select for DB errors
                logger.error(f"Query execution failed as expected: {e}")
            except Exception as e: # Catch any other unexpected errors during this specific query test
                logger.error(f"An unexpected error occurred during test execution for query '{query[:100]}...': {e}", exc_info=True)

        logger.info("\n--- QueryExecutor Test Complete ---")

    except ConnectionError as e: # Catch database connection errors during setup (db_connector init)
        logger.error(f"QueryExecutor test failed: Database connection error during setup: {e}")
    except Exception as e: # Catch any other errors during the overall test setup or run
        logger.error(f"QueryExecutor test failed: An unexpected error occurred during test setup or run: {e}", exc_info=True)
    # --- End Main Test Execution Block ---