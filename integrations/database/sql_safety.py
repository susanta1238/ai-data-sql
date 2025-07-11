# integrations/database/sql_safety.py
import logging
import sqlglot
import sqlglot.errors
import sqlglot.expressions # Import expressions module to check types
import sys

# You might want to import the SchemaLoader or DatabaseConnector here
# if your validation logic needs database context (e.g., checking if tables exist)
# from .schema_loader import SchemaLoader
# from .db_connector import DatabaseConnector

logger = logging.getLogger(__name__)

class SQLSafetyChecker:
    """
    Validates and sanitizes generated SQL queries for safety.
    """
    def __init__(self, config=None): # Config made optional if not strictly needed for init
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


    def is_safe(self, sql_query: str, schema_info=None) -> bool:
        """
        Performs safety checks on the generated SQL query.

        Args:
            sql_query (str): The SQL query string to check.
            schema_info (dict, optional): Dictionary containing database schema.
                                           Used for context-aware validation.

        Returns:
            bool: True if the query is considered safe, False otherwise.
        """
        if not sql_query or not isinstance(sql_query, str) or not sql_query.strip():
            logger.warning("SQL safety check failed: Empty or invalid input.")
            return False

        query_stripped = sql_query.strip()

        # --- 1. Basic Keyword Check (Quick Fail for obvious dangers) ---
        # Case-insensitive check for specific highly dangerous keywords/patterns
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
                 # This should be rare due to prior strip and keyword checks, but good safety net.
                 logger.warning("SQL safety check failed: No executable statement found after parsing.")
                 return False

            # Get the single parsed statement
            parsed_query = expressions[0]

            # --- FIX: Check if the parsed expression is None or not a standard expression type ---
            # This handles cases where the input was only comments/whitespace
            if parsed_query is None or not isinstance(parsed_query, sqlglot.expressions.Expression):
                 logger.warning(f"SQL safety check failed: Parsed content is not a valid SQL expression type ({type(parsed_query).__name__ if parsed_query is not None else 'None'}).")
                 return False


            # Now we know parsed_query is a valid sqlglot Expression instance
            statement_type = parsed_query.__class__.__name__.upper()
            logger.debug(f"Checking single statement (type: {statement_type}): {parsed_query.sql(dialect='tsql')[:100]}...")

            # Check the type of statement (only allow SELECT)
            is_allowed_type = any(isinstance(parsed_query, allowed_class) for allowed_class in self.ALLOWED_STATEMENT_CLASSES.values())

            if not is_allowed_type:
                # Get the name of the actual statement type for logging
                allowed_names = list(self.ALLOWED_STATEMENT_CLASSES.keys())
                logger.warning(f"SQL safety check failed: Disallowed statement type '{statement_type}'. Only {allowed_names} are allowed.")
                return False

            # Additional checks for SELECT statements

            if isinstance(parsed_query, sqlglot.expressions.Select):
                # Check for missing TOP/LIMIT if required (ensures result set size is limited)
                # Check for the Limit expression node (sqlglot normalizes TOP/LIMIT to Limit)
                limit_node = parsed_query.find(sqlglot.expressions.Limit)

                is_limited = False
                limit_value = float('inf') # Assume no explicit limit value
                if limit_node and limit_node.this and limit_node.this.name:
                    try:
                        # Attempt to parse the limit value as an integer
                        limit_value = int(limit_node.this.name)
                        is_limited = True
                        logger.debug(f"Found LIMIT/TOP clause with value: {limit_value}")
                    except (ValueError, TypeError):
                         logger.warning("SQL safety check: Could not parse LIMIT/TOP value as integer.")
                         return False # Treat unparsable limit as unsafe


                # Decide if *all* SELECT queries *must* have a limit (e.g., require TOP/LIMIT)
                # If you uncomment this, queries without TOP/LIMIT will fail.
                # if not is_limited:
                #      logger.warning("SQL safety check failed: SELECT query is not explicitly limited (missing TOP/LIMIT).")
                #      return False

                # Enforce a maximum number of rows if TOP/LIMIT is present
                if is_limited and limit_value > self.MAX_ROWS_FETCH:
                     logger.warning(f"SQL safety check failed: LIMIT/TOP value ({limit_value}) exceeds max allowed ({self.MAX_ROWS_FETCH}). Max allowed: {self.MAX_ROWS_FETCH}")
                     return False

                # Check for UNION/UNION ALL using parser
                if parsed_query.find(sqlglot.expressions.Union):
                     logger.warning("SQL safety check failed: UNION or UNION ALL found. These are disallowed within a single SELECT statement.")
                     return False

                # Check for excessive joins (example - requires AST traversal)
                # joins_count = 0
                # for node, parent, key in parsed_query.walk():
                #    if isinstance(node, sqlglot.expressions.Join):
                #        joins_count += 1
                # if joins_count > self.MAX_JOINS:
                #     logger.warning(f"SQL safety check failed: Excessive number of joins ({joins_count}). Max allowed: {self.MAX_JOINS}.")
                #     return False


            # Context-aware checks (Optional, requires schema_info)
            # If schema_info is provided and you want to enable schema checks:
            # if schema_info:
            #    try:
            #        if not self._validate_schema_elements(parsed_query, schema_info):
            #             return False # _validate_schema_elements logs specific failure
            #    except Exception as e:
            #        logger.error(f"An error occurred during schema validation: {e}", exc_info=True)
            #        return False # Fail if schema validation logic itself errors


        except sqlglot.errors.ParseError as e:
            logger.warning(f"SQL safety check failed: Failed to parse query (Syntax Error): {e}")
            return False
        except Exception as e:
            # Catch any other unexpected errors during parsing or AST analysis
            logger.error(f"An unexpected error occurred during SQL safety check parsing or AST analysis: {e}", exc_info=True)
            return False


        # If all checks pass
        logger.debug("SQL query passed safety checks.")
        return True

    # Optional: Helper method for complex schema validation (requires implementation)
    # def _validate_schema_elements(self, parsed_query, schema_info) -> bool:
    #    """Traverses parsed query AST and checks if tables/columns exist in schema_info."""
    #    logger.debug("Schema validation is not fully implemented or skipped.")
    #    return True


# Example usage (Test Block) - Run with `python -m integrations.database.sql_safety`
if __name__ == "__main__":
    # Setup logging
    try:
        from utils.logging_config import setup_logging
        # Need a dummy config object for logging setup
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
        logger = logging.getLogger(__name__)


    try:
        logger.info("--- Testing SQLSafetyChecker ---")
        # Load configuration (not strictly needed for basic checks, but good practice)
        # Import here for test block to avoid issues with main app import order
        from utils.config_loader import load_config
        app_config = load_config()

        # Initialize the checker
        safety_checker = SQLSafetyChecker(app_config)

        # Define test cases
        test_queries = [
            "SELECT * FROM Customers WHERE City = 'London';",                     # Safe SELECT (with semicolon)
            "SELECT TOP 10 FirstName, LastName FROM Employees;",                 # Safe SELECT with TOP
            "SELECT COUNT(*) FROM Orders",                                       # Safe SELECT aggregate
            "SELECT c.Name, o.OrderDate FROM Customers c JOIN Orders o ON c.Id = o.CustomerId;", # Safe SELECT with JOIN
            "DROP TABLE SensitiveData;",                                         # UNSAFE - DROP
            "DELETE FROM Users WHERE IsActive = 0",                              # UNSAFE - DELETE
            "SELECT * FROM Products; -- DROP DATABASE;",                         # UNSAFE - Multiple statements or Keyword
            "SELECT * FROM Products; DROP DATABASE;",                            # UNSAFE - Multiple statements (parser error likely)
            "UPDATE Settings SET Value = 'Admin' WHERE Key = 'UserRole'",       # UNSAFE - UPDATE
            "EXEC sp_configure 'show advanced options', 1",                      # UNSAFE - EXEC sp_
            "SELECT 1; SELECT 2;",                                               # UNSAFE - Multiple statements
            "SELECT * FROM Users WHERE UserId = 1 OR 1=1 --';",                  # Safe (comment inside string literal, parser handles)
            "SELECT column FROM table WHERE id = 'value%';",                    # Safe (contains %)
            "SELECT * FROM Users WHERE Name = 'O''Malley';",                     # Safe (contains quoted string)
            "SELECT * FROM NonExistentTable;",                                   # Safe SYNTAX, but semantically invalid (Schema check needed - will pass here)
            "SELECT * FROM table1 UNION SELECT * FROM table2;",                  # UNSAFE - UNION (caught by parser find)
            "SELECT * FROM table1 UNION ALL SELECT * FROM table2",               # UNSAFE - UNION ALL (caught by parser find)
            # "SELECT * FROM VeryLargeTable;",                                     # Potentially unsafe - No Limit (Fails if you uncomment limit check)
            "SELECT TOP 10000 * FROM AnotherTable;",                             # Potentially unsafe - TOP exceeds limit (Fails if you uncomment TOP limit check)
            "", # Empty query
            "   \n  -- comment only \n ", # Whitespace and comments only
            "/* This is a block comment */", # Block comment only
            "/* This is a block comment */ SELECT 1;", # Safe SELECT with leading comment
        ]

        # Run tests
        for query in test_queries:
            logger.info(f"\nChecking query: '{query}'")
            # Pass schema_info if you implement _validate_schema_elements
            # is_safe = safety_checker.is_safe(query, schema_info={'dbo.ProfileData': [...]}) # Example passing schema
            is_safe = safety_checker.is_safe(query)
            logger.info(f"Result: {'SAFE' if is_safe else 'UNSAFE'}")

        logger.info("\n--- SQLSafetyChecker Test Complete ---")

    except Exception as e:
        logger.error(f"SQLSafetyChecker test failed: An unexpected error occurred: {e}", exc_info=True)