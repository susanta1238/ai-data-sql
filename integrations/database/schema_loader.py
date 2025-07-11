# integrations/database/schema_loader.py
import logging
# Assuming DatabaseConnector is implemented in db_connector.py in the same directory
from .db_connector import DatabaseConnector
from utils.config_loader import load_config # Needed for the test block
import sys

logger = logging.getLogger(__name__)

class SchemaLoader:
    """
    Loads schema information (tables, columns) from the database.
    """
    def __init__(self, db_connector: DatabaseConnector):
        """
        Initializes with a DatabaseConnector instance.
        """
        self.db_connector = db_connector
        logger.info("SchemaLoader initialized.")

    def get_table_names(self):
        """
        Retrieves a list of all table names in the database.
        """
        conn = None
        try:
            conn = self.db_connector.get_connection()
            cursor = conn.cursor()
            # Query INFORMATION_SCHEMA.TABLES for user tables
            # TABLE_TYPE = 'BASE TABLE' is standard for user tables
            # Adjust TABLE_SCHEMA if you only want tables from a specific schema (e.g., 'dbo')
            query = "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_SCHEMA, TABLE_NAME;"
            cursor.execute(query)
            table_names = [f"{row.TABLE_SCHEMA}.{row.TABLE_NAME}" for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(table_names)} table names.")
            return table_names
        except Exception as e:
            logger.error("Error retrieving table names from database.", exc_info=True)
            raise RuntimeError("Failed to load table names.") from e
        finally:
            if conn:
                self.db_connector.close_connection(conn)

    def get_column_info(self, table_name):
        """
        Retrieves column details (name, data type) for a specific table.
        Input table_name should include schema (e.g., 'dbo.Customers').
        """
        conn = None
        try:
            conn = self.db_connector.get_connection()
            cursor = conn.cursor()

            # Parse schema and table name
            parts = table_name.split('.')
            if len(parts) != 2:
                 raise ValueError(f"Invalid table name format: '{table_name}'. Expected 'schema.table'.")
            schema, name = parts

            # Query INFORMATION_SCHEMA.COLUMNS
            # Parameterize the query to prevent SQL injection if table_name comes from user input (though here it comes from get_table_names)
            query = """
            SELECT COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION;
            """
            cursor.execute(query, (schema, name))

            # Format column info as a list of strings like "column_name (data_type)"
            column_info = [f"{row.COLUMN_NAME} ({row.DATA_TYPE})" for row in cursor.fetchall()]
            logger.debug(f"Retrieved column info for {table_name}: {column_info}")
            return column_info
        except ValueError:
             raise # Re-raise the handled value error
        except Exception as e:
            logger.error(f"Error retrieving column info for table '{table_name}'.", exc_info=True)
            raise RuntimeError(f"Failed to load column info for table '{table_name}'.") from e
        finally:
            if conn:
                self.db_connector.close_connection(conn)

    def get_full_schema(self):
        """
        Retrieves schema information for all tables and their columns.
        Returns a dictionary where keys are table names and values are lists of column info strings.
        """
        logger.info("Starting full schema load.")
        all_schema_info = {}
        try:
            table_names = self.get_table_names()
            for table_name in table_names:
                try:
                    column_info = self.get_column_info(table_name)
                    all_schema_info[table_name] = column_info
                except RuntimeError:
                    # Log error in get_column_info, continue with other tables
                    logger.warning(f"Skipping schema for table '{table_name}' due to error.")
                    pass # Continue to the next table if loading one table failed
            logger.info(f"Full schema load complete for {len(all_schema_info)} tables.")
            return all_schema_info
        except RuntimeError:
             # get_table_names already logged the error
             raise # Re-raise the initial failure
        except Exception as e:
             logger.error("An unexpected error occurred during full schema load.", exc_info=True)
             raise RuntimeError("An unexpected error occurred during full schema load.") from e


# Example usage (Test Block)
if __name__ == "__main__":
    # Setup logging before loading config and using the connector/schema loader
    try:
        from utils.logging_config import setup_logging
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' } # Use DEBUG level for test output
        setup_logging(DummyConfigForLoggingTest())
        logger.info("Temporary logging setup for schema_loader test.")
    except ImportError:
        print("Could not import utils.logging_config. Logging output might be basic.", file=sys.stderr)
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)


    try:
        logger.info("--- Testing SchemaLoader ---")
        # Load configuration
        app_config = load_config()

        # Initialize the DatabaseConnector (SchemaLoader needs an instance)
        db_connector = DatabaseConnector(app_config)

        # Initialize the SchemaLoader
        schema_loader = SchemaLoader(db_connector)

        # Test getting table names
        logger.info("Attempting to get table names...")
        table_list = schema_loader.get_table_names()
        logger.info(f"Tables found: {table_list}")

        # Test getting schema for a specific table (replace with a table from your DB)
        if table_list:
             first_table = table_list[0] # Use the first table found
             logger.info(f"Attempting to get schema for table: {first_table}")
             column_info_list = schema_loader.get_column_info(first_table)
             logger.info(f"Schema for {first_table}: {column_info_list}")
        else:
             logger.warning("No tables found to test get_column_info.")


        # Test getting full schema
        logger.info("Attempting to get full schema...")
        full_schema_dict = schema_loader.get_full_schema()
        logger.info(f"Full schema loaded for {len(full_schema_dict)} tables.")
        # Print a snippet of the full schema for verification
        for i, (table, columns) in enumerate(full_schema_dict.items()):
            logger.info(f"  Table '{table}': {columns}")
            if i >= 4: # Print info for max 5 tables to keep output manageable
                logger.info("  ...")
                break


        logger.info("--- SchemaLoader Test Complete ---")

    except (ConnectionError, RuntimeError, ValueError) as e:
        logger.error(f"SchemaLoader test failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"SchemaLoader test failed: An unexpected error occurred: {e}", exc_info=True)