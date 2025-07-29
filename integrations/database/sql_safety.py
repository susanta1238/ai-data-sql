# integrations/database/sql_safety.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
# Add this line to help debug if this specific file version is being loaded.
# If this message doesn't appear in your terminal logs during startup or test,
# it means an older version of this file is being used due to caching or other environmental issue.
print("DEBUG: Loading integrations/database/sql_safety.py - Version Check 25") # Increment version
# --- END ADDITION ---


import logging
import sqlglot
import sqlglot.errors
import sqlglot.expressions # Import expressions module to check types and walk AST
import sys # Import sys for test block
from typing import List, Dict, Any, Optional # Import typing for type hints


logger = logging.getLogger(__name__) # Get logger at module level

class SQLSafetyChecker:
    """
    Validates and sanitizes generated SQL queries for safety.
    Includes checks for disallowed keywords, multiple statements,
    disallowed statement types, TOP/LIMIT limits, and (optionally) schema validation.
    """
    def __init__(self, config=None):
        """
        Initializes the checker with application configuration if needed.
        """
        self.config = config # Store config if validation rules depend on it
        logger.info("SQLSafetyChecker initialized.")

        # Define disallowed SQL patterns or keywords (used for quick initial check)
        # Rely more on the parser for statement types (DROP, DELETE, UPDATE, etc.)
        # Rely on multi-statement check for ';' and comments.
        # Focus this list on dangerous functions or patterns unlikely in a valid SELECT.
        self.DISALLOWED_KEYWORDS = [
            "xp_", "sp_", # Stored procedures, especially system ones like xp_cmdshell
            "WAITFOR DELAY", "BENCHMARK", # Time-based attacks or denial-of-service
            # You could add other specific dangerous functions here if known for your DB
            # e.g., "BULK INSERT"
        ]

        # Define statement types allowed (using sqlglot expression class names)
        # Map string names to sqlglot classes for robust type checking
        self.ALLOWED_STATEMENT_CLASSES = {
            "SELECT": sqlglot.expressions.Select
        } # Only allow SELECT queries by default

        # Define limits (e.g., max joins, max rows using LIMIT/TOP)
        self.MAX_JOINS = 5 # Example limit (requires AST traversal implementation)
        self.MAX_ROWS_FETCH = 1000 # Recommend limiting results if TOP/LIMIT is used


    def is_safe(self, sql_query: str, schema_info: Optional[Dict[str, List[str]]] = None) -> bool:
        """
        Performs safety checks on the generated SQL query.

        Args:
            sql_query (str): The SQL query string to check.
            schema_info (dict, optional): Dictionary containing database schema.
                                           Expected format: { 'schema.table': ['column1 (type)', 'column2 (type)'], ... }
                                           Used for context-aware validation (e.g., checking if tables/columns exist).
                                           Pass this from the SchemaLoader results.

        Returns:
            bool: True if the query is considered safe, False otherwise.
        """
        if not sql_query or not isinstance(sql_query, str) or not sql_query.strip():
            logger.warning("SQL safety check failed: Empty or invalid input.")
            return False

        query_stripped = sql_query.strip()

        # --- 1. Basic Keyword Check (Quick Fail for obvious dangers) ---
        query_upper = query_stripped.upper()
        for keyword in self.DISALLOWED_KEYWORDS:
            if keyword in query_upper:
                logger.warning(f"SQL safety check failed: Disallowed keyword '{keyword}' found.")
                return False


        # --- 2. SQL Parser-based Validation (Recommended & more robust) ---
        try:
            # Parse all statements using sqlglot. Specify the SQL dialect ("tsql" for SQL Server).
            # ErrorLevel.RAISE will raise exceptions for syntax errors during parsing.
            # This also correctly handles comments and statement separation.
            expressions = sqlglot.parse(query_stripped, dialect="tsql", error_level=sqlglot.ErrorLevel.RAISE)
            logger.debug(f"SQL query parsed into {len(expressions)} statements.")

            # Check for multiple statements
            if len(expressions) > 1:
                 logger.warning(f"SQL safety check failed: Multiple statements found.")
                 return False

            # Handle case where parse returns empty list (e.g., only comments or whitespace, now caught earlier)
            if not expressions:
                 logger.warning("SQL safety check failed: No executable statement found after parsing.")
                 return False # This should be rare due to prior strip and keyword checks


            # Get the single parsed statement
            parsed_query = expressions[0]

            # Check if the parsed expression is None or not a standard expression type (handles comments-only etc.)
            if parsed_query is None or not isinstance(parsed_query, sqlglot.expressions.Expression):
                 logger.warning(f"SQL safety check failed: Parsed content is not a valid SQL expression type ({type(parsed_query).__name__ if parsed_query is not None else 'None'}).")
                 return False


            # Now we know parsed_query is a valid sqlglot Expression instance
            statement_type = parsed_query.__class__.__name__.upper()
            logger.debug(f"Checking single statement (type: {statement_type}): {parsed_query.sql(dialect='tsql')[:100]}...")

            # Check the type of statement (only allow SELECT)
            is_allowed_type = any(isinstance(parsed_query, allowed_class) for allowed_class in self.ALLOWED_STATEMENT_CLASSES.values())

            if not is_allowed_type:
                actual_statement_type_name = statement_type # Already uppercased
                allowed_names = list(self.ALLOWED_STATEMENT_CLASSES.keys())
                logger.warning(f"SQL safety check failed: Disallowed statement type '{actual_statement_type_name}'. Only {allowed_names} are allowed.")
                return False

            # Additional checks for SELECT statements

            if isinstance(parsed_query, sqlglot.expressions.Select):
                limit_node = parsed_query.find(sqlglot.expressions.Limit)
                is_limited = False
                limit_value = float('inf')
                if limit_node and limit_node.this and limit_node.this.name:
                    try:
                        limit_value = int(limit_node.this.name)
                        is_limited = True
                        logger.debug(f"Found LIMIT/TOP clause with value: {limit_value}")
                    except (ValueError, TypeError):
                         logger.warning("SQL safety check: Could not parse LIMIT/TOP value as integer.")
                         return False

                if is_limited and limit_value > self.MAX_ROWS_FETCH:
                     logger.warning(f"SQL safety check failed: LIMIT/TOP value ({limit_value}) exceeds max allowed ({self.MAX_ROWS_FETCH}). Max allowed: {self.MAX_ROWS_FETCH}")
                     return False

                if parsed_query.find(sqlglot.expressions.Union):
                     logger.warning("SQL safety check failed: UNION or UNION ALL found. These are disallowed within a single SELECT statement.")
                     return False

                # Check for excessive joins (example - requires AST traversal implementation)
                # joins_count = 0
                # for walk_item in parsed_query.walk(): # Use the correct walk iteration
                #    node = walk_item[0]
                #    if isinstance(node, sqlglot.expressions.Join):
                #        joins_count += 1
                # if joins_count > self.MAX_JOINS:
                #     logger.warning(f"SQL safety check failed: Excessive number of joins ({joins_count}). Max allowed: {self.MAX_JOINS}.")
                #     return False

                # --- Implement Schema Validation ---
                if schema_info:
                   logger.debug("Performing schema validation...")
                   try:
                       # Call the helper method to validate elements against schema
                       # Pass the parsed query expression itself
                       if not self._validate_schema_elements(parsed_query, schema_info):
                           logger.warning("SQL safety check failed: Schema validation failed.")
                           return False
                       logger.debug("Schema validation passed.")
                   except Exception as e:
                       logger.error(f"An unexpected error occurred during schema validation: {e}", exc_info=True)
                       return False
                else:
                   logger.warning("Schema validation skipped: schema_info not provided.")


        except sqlglot.errors.ParseError as e:
            logger.warning(f"SQL safety check failed: Failed to parse query (Syntax Error): {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during SQL safety check parsing or AST analysis: {e}", exc_info=True)
            return False

        logger.debug("SQL query passed safety checks.")
        return True

    # --- Helper method for schema validation (Corrected walk() iteration) ---
    def _validate_schema_elements(self, parsed_query: sqlglot.expressions.Expression, schema_info: dict) -> bool:
       """
       Traverses parsed query AST and checks if tables and columns exist in schema_info.
       Correctly iterates over walk() results. Checks tables and basic column names.
       Requires schema_info format: { 'schema.table': ['column1 (type)', 'column2 (type)'], ... }
       Returns True if all referenced tables/columns are found, False otherwise.

       NOTE: This provides basic validation. Full scope-aware column validation with aliases/subqueries
       is complex and not fully implemented here. This checks if names exist *somewhere* in the schema.
       """
       logger.debug("Starting schema element validation (tables and basic columns)...")
       is_valid = True

       # Pre-process schema_info for efficient lookups
       schema_table_names = {name.lower() for name in schema_info.keys()}
       # Store lowercase column names in a set for quick lookup
       all_schema_column_names = {col.split(' ')[0].lower() for cols in schema_info.values() for col in cols}

       referenced_column_names_in_query = set() # Track column names seen in the query


       # Walk the Abstract Syntax Tree (AST) of the parsed query
       # Iterate over walk() results.
       # walk() yields (node, parent, key) for children, but (root_node,) for the root.
       for walk_item in parsed_query.walk():
           # --- FIX: Safely unpack the walk_item and check type ---
           # Check if the yielded item is a tuple before trying to get parent/key
           node = walk_item[0] # The node is always the first item

           # Safely get parent and key; they might not be present if walk_item is not a tuple
           # Check if walk_item is a tuple before accessing elements beyond index 0
           parent = walk_item[1] if isinstance(walk_item, tuple) and len(walk_item) > 1 else None
           key = walk_item[2] if isinstance(walk_item, tuple) and len(walk_item) > 2 else None

           # --- END FIX ---


           # Only process if the node is a sqlglot Expression (filter out potential other types from walk)
           if not isinstance(node, sqlglot.expressions.Expression):
               logger.debug(f"Skipping non-expression node in walk: {type(node)}")
               continue


           # Check for table references (sqlglot.expressions.Table nodes)
           # Check if node is a Table expression before accessing its attributes
           if isinstance(node, sqlglot.expressions.Table):
               # Extract the full table name (schema.table)
               schema_name = 'dbo' # Default if schema not explicitly in query
               # Safely get the schema name from node.this if it's an Identifier and has a db attribute
               if isinstance(node.this, sqlglot.expressions.Identifier) and hasattr(node.this, 'db') and node.this.db:
                    schema_name = node.this.db.name
               table_name_simple = node.this.name if isinstance(node.this, sqlglot.expressions.Identifier) else None # Get the identifier name

               if table_name_simple: # Ensure we got a table name
                   full_name_check = f"{schema_name}.{table_name_simple}"

                   # Check if the table exists in the provided schema_info set (case-insensitive)
                   if full_name_check.lower() not in schema_table_names:
                       logger.warning(f"Schema validation failed: Table '{full_name_check}' referenced in query not found in schema.")
                       is_valid = False # Mark as invalid, but continue checking other elements
               else:
                   # Handle cases where node is Table but .this is not a simple Identifier (less common?)
                   logger.warning(f"Schema validation encountered unexpected Table node structure: {type(node.this)}")
                   is_valid = False


           # Check for column references (sqlglot.expressions.Column nodes)
           # Check if node is a Column expression before accessing its attributes
           if isinstance(node, sqlglot.expressions.Column):
               # Column names are identifiers
               column_name_simple = node.this.name if isinstance(node.this, sqlglot.expressions.Identifier) else None

               if column_name_simple: # Ensure we got a column name
                   # Store the referenced column name (simplified check for now)
                   referenced_column_names_in_query.add(column_name_simple.lower())

                   # --- Basic Column Check: Check if column name exists in the schema ---
                   # This is a quick check, but not scope-aware.
                   if column_name_simple.lower() not in all_schema_column_names:
                       logger.warning(f"Schema validation failed: Column '{column_name_simple}' referenced in query not found in *any* table in schema.")
                       is_valid = False # Mark as invalid


                   # --- TODO: Advanced Column Check Logic (Requires scope awareness) ---
                   # This is where you'd check if the column *belongs* to the table/alias it's referenced with.
                   # This requires analyzing the query's FROM/JOIN/Alias structure.


               else:
                   # Handle cases where node is Column but .this is not a simple Identifier
                    logger.warning(f"Schema validation encountered unexpected Column node structure: {type(node.this)}")
                    is_valid = False


       # --- Final check for validity ---
       if not is_valid:
           logger.warning("Schema validation failed due to missing tables or columns.")
           return False # Return False if any element was invalid

       logger.debug("Schema validation completed successfully (tables and basic column check).")
       return True # Return True if all checks passed

# ... (rest of the test block remains the same, including sample_schema_info) ...

# Example usage (Test Block) - Run with `python -m integrations.database.sql_safety`
if __name__ == "__main__":
    # Setup logging
    try:
        from utils.logging_config import setup_logging
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' }
                 self.app = {}
                 self.openai = {}
                 self.database = {}
        setup_logging(DummyConfigForLoggingTest())
        logger.info("Temporary logging setup for sql_safety test.")
    except ImportError:
        print("Could not import utils.logging_config. Logging output might be basic.", file=sys.stderr)
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG)
        # Re-get logger here if basicConfig was needed
        logger = logging.getLogger(__name__)


    try:
        logger.info("--- Testing SQLSafetyChecker with Schema Validation ---")
        # Load configuration (not strictly needed for basic checks, but good practice)
        from utils.config_loader import load_config # Import here for test block
        app_config = load_config()

        # Initialize the checker
        safety_checker = SQLSafetyChecker(app_config)

        # --- Define a Sample SchemaInfo dictionary for testing ---
        # This MIMICS the output of your SchemaLoader for dbo.ProfileData
        # It must match the format: { 'schema.table': ['column1 (type)', 'column2 (type)'], ... }
        # Based on the CREATE TABLE provided earlier (all columns):
        sample_schema_info = {
            'dbo.ProfileData': [
                'ProfileId (bigint)', 'created_by (varchar)', 'created_on (datetime)', 'data_source (varchar)',
                'person_profile_headline (varchar)', 'job_last_updated (nvarchar)', 'job_start_date (nvarchar)',
                'job_summary (nvarchar)', 'job_title (varchar)', 'job_title_role (varchar)', 'job_title_sub_role (varchar)',
                'job_title_levels (varchar)', 'organization_domain (varchar)', 'organization_email (varchar)',
                'organization_email_status (varchar)', 'organization_email_validation_source (varchar)',
                'organization_email_validation_date (date)', 'organization_email_last_opened_date (datetime)',
                'organization_email_last_clicked_date (datetime)', 'organization_facebook_url (varchar)',
                'organization_founded_year (varchar)', 'organization_location_address_line_2 (varchar)',
                'organization_location_city (varchar)', 'organization_location_city_state_country (varchar)',
                'organization_location_continent (varchar)', 'organization_location_country (varchar)',
                'organization_location_geo_code (varchar)', 'organization_location_postal_code (varchar)',
                'organization_location_region (varchar)', 'organization_location_state_country (varchar)',
                'organization_location_state_address (varchar)', 'organization_industries (varchar)',
                'organization_linkedin_id (varchar)', 'organization_linkedin_url (varchar)', 'organization_name (varchar)',
                'organization_phone (varchar)', 'organization_phone_status (varchar)', 'organization_phone_validation_source (varchar)',
                'organization_phone_validation_date (datetime)', 'organization_phone_last_communicated_date (datetime)',
                'organization_size (varchar)', 'organization_twitter_url (varchar)', 'person_birth_date (varchar)',
                'person_birth_year (varchar)', 'person_email (varchar)', 'person_email_status (varchar)',
                'person_email_validation_source (varchar)', 'person_email_validation_date (datetime)',
                'person_email_last_opened_date (datetime)', 'person_email_last_clicked_date (datetime)',
                'person_facebook_id (varchar)', 'person_facebook_url (varchar)', 'person_facebook_url_status (varchar)',
                'person_facebook_url_validation_source (varchar)', 'person_facebook_url_validation_date (datetime)',
                'person_facebook_username (varchar)', 'person_first_name (varchar)', 'person_gender (varchar)',
                'person_github_url (varchar)', 'person_github_username (varchar)', 'person_industries (varchar)',
                'person_inferred_salary (varchar)', 'person_inferred_years_experience (varchar)', 'person_interest (varchar)',
                'person_last_name (varchar)', 'person_linkedin_connections (varchar)', 'person_linkedin_id (varchar)',
                'person_linkedin_url (varchar)', 'person_linkedin_url_status (varchar)',
                'person_linkedin_url_validation_source (varchar)', 'person_linkedin_url_validation_date (datetime)',
                'person_linkedin_url_last_communicated_date (datetime)', 'person_linkedin_username (varchar)',
                'person_location_address_line_2 (varchar)', 'person_location_city (varchar)', 'person_location_city_status_country (varchar)',
                'person_location_continent (varchar)', 'person_location_country (varchar)', 'person_location_geo_code (varchar)',
                'person_location_last_updated (varchar)', 'person_location_state_country (varchar)',
                'person_location_postal_code (varchar)', 'person_location_state (varchar)', 'person_location_street_address (varchar)',
                'person_middle_initial (varchar)', 'person_middle_name (varchar)', 'person_mobile (varchar)',
                'person_mobile_status (varchar)', 'person_mobile_validation_source (varchar)', 'person_mobile_validation_date (datetime)',
                'person_mobile_last_communicated_date (datetime)', 'person_full_name (varchar)', 'person_phone (varchar)',
                'person_phone_status (varchar)', 'person_phone_validation_source (varchar)', 'person_phone_validation_date (datetime)',
                'person_phone_last_communicated_date (datetime)', 'person_raw_number (varchar)', 'person_skills (nvarchar)',
                'person_twitter_url (varchar)', 'person_twitter_url_status (varchar)',
                'person_twitter_url_validation_source (varchar)', 'person_twitter_url_validation_date (datetime)',
                'person_twitter_username (varchar)', 'search_tags (varchar)', 'updated_by (varchar)', 'updated_on (datetime)',
                'person_photo_url (nvarchar)', 'organization_phone_sanitized (varchar)', 'ConfidenceScore (int)',
                'PersonEmailScore (int)', 'OrganisationEmailScore (int)', 'PersonPhoneScore (int)', 'PersonMobileScore (int)',
                'OrganisationPhoneScore (int)', 'PersonLinkedInScore (int)', 'DataBatch (varchar)', 'person_email_opened (bit)',
                'person_email_clicked (bit)', 'IsUnsubscribed (bit)', 'Unsubscribed_on (datetime)', 'person_twitter_followers (int)',
                'person_twitter_createdon (date)', 'organization_email_clicked (bit)', 'organization_email_opened (bit)'
            ],
            # Add other tables/columns if your database has them and they are used in queries
            # Example: 'dbo.Orders': ['OrderId (int)', 'OrderDate (datetime)', 'CustomerId (int)']
            # Example: 'dbo.Customers': ['CustomerId (int)', 'Name (varchar)']
        }
        # Add another dummy table to test cross-table validation failure
        sample_schema_info['dbo.DummyTable'] = ['DummyColumnA (int)', 'DummyColumnB (varchar)']

        logger.debug(f"Using sample_schema_info with {len(sample_schema_info)} tables.")
        total_cols = sum(len(cols) for cols in sample_schema_info.values())
        logger.debug(f"Total columns in sample_schema_info: {total_cols}")


        # Define test cases
        test_queries = [
            # Safe queries that should pass all checks including schema validation (assuming schema_info is correct)
            "SELECT ProfileId, person_full_name FROM dbo.ProfileData WHERE person_location_country = 'United States';", # Valid table, valid columns
            "SELECT TOP 10 person_full_name, job_title FROM dbo.ProfileData;", # Valid table, valid columns
            "SELECT COUNT(*) FROM dbo.ProfileData WHERE organization_size = 'Large'", # Valid table, valid column, aggregate
            "SELECT person_full_name, organization_name FROM dbo.ProfileData WHERE person_location_state = 'California';",
            "SELECT TOP 20 ProfileId, person_location_city FROM dbo.ProfileData WHERE person_location_city = 'New York';",
            "SELECT COUNT(*) FROM dbo.ProfileData WHERE YEAR(updated_on) = 2024;", # Using YEAR function on datetime column
            "SELECT TOP 50 job_title, organization_name FROM dbo.ProfileData ORDER BY ProfileId;",
            "SELECT person_full_name, person_email FROM dbo.ProfileData WHERE person_email_status = 'valid';",
            "SELECT person_full_name, person_phone, person_mobile FROM dbo.ProfileData WHERE person_full_name = 'John Doe';", # Querying contact details
            "SELECT ProfileId, organization_domain, organization_size FROM dbo.ProfileData WHERE organization_industries LIKE '%Technology%';", # Using LIKE filter on industries

            # Queries that should FAIL schema validation (due to missing tables or columns in sample_schema_info)
            "SELECT * FROM NonExistentTable;",                                   # Missing table
            "SELECT * FROM dbo.ProfileData WHERE NonExistentColumn = 'value';",  # Missing column (currently only tables are checked robustly)
            "SELECT Name FROM dbo.ProfileData;",                                  # Missing column 'Name' (assuming 'Name' is not in ProfileData columns)
            "SELECT c.Name FROM Customers c;",                                    # Missing table 'Customers' and column 'Name'
            "SELECT * FROM dbo.ProfileData p JOIN dbo.OtherTable o ON p.Id = o.ProfileId", # Missing 'OtherTable' (assuming not in sample schema)
            "SELECT * FROM dbo.ProfileData WHERE person_location = 'Middle East';", # Missing column 'person_location' (assuming only city, state, country, continent etc. exist)
            "SELECT DummyColumnA FROM dbo.DummyTable WHERE DummyColumnB = 'test';", # Valid syntax, should pass based on sample_schema_info

            # Unsafe syntax/type queries (should be caught by existing checks before schema validation)
            "DROP TABLE SensitiveData;",
            "SELECT 1; SELECT 2;",
            "EXEC sp_configure;",

            # Queries that might parse oddly but still represent SELECTs
             "SELECT 1+1;", # Simple arithmetic expression
             "SELECT YEAR(GETDATE());", # Function call
             "SELECT * FROM (SELECT 1) as T;", # Subquery

        ]

        # Run tests
        for query in test_queries:
            # Truncate query for log if very long
            log_query = query if len(query) < 100 else query[:97] + '...'
            logger.info(f"\nChecking query: '{log_query}'")
            # --- Pass the sample_schema_info to is_safe ---
            is_safe = safety_checker.is_safe(query, schema_info=sample_schema_info)
            logger.info(f"Result: {'SAFE' if is_safe else 'UNSAFE'}")

        logger.info("\n--- SQLSafetyChecker Test Complete ---")

    except Exception as e:
        logger.error(f"SQLSafetyChecker test failed: An unexpected error occurred: {e}", exc_info=True)