# integrations/database/db_connector.py
import pyodbc
import logging
from utils.config_loader import load_config # Import the config loader
import sys

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Handles connections to the SQL Server database.
    """
    def __init__(self, config):
        """
        Initializes the connector with application configuration.
        """
        self.config = config.database
        self.connection_string = config.build_db_connection_string()
        logger.info(f"DatabaseConnector initialized with connection string: {self.connection_string.split('PWD=')[0]}...") # Log sensitive string safely

        # Basic check to ensure driver exists (pyodbc.drivers() returns a list)
        required_driver = self.config.get('odbc_driver')
        if required_driver and required_driver.strip('{}') not in [d.strip('{}') for d in pyodbc.drivers()]:
             logger.error(f"ODBC Driver '{required_driver}' not found. Available drivers: {pyodbc.drivers()}")
             # In production, you might raise an exception or exit here
             # raise EnvironmentError(f"Required ODBC Driver '{required_driver}' not found.")


    def get_connection(self):
        """
        Establishes and returns a new database connection.
        Handles connection errors.
        """
        try:
            # Set connection timeout (optional but recommended)
            # pyodbc.connect supports timeout parameter
            conn = pyodbc.connect(self.connection_string, timeout=30) # 30 seconds timeout
            logger.info("Database connection established successfully.")
            return conn
        except pyodbc.Error as ex:
            sqlstate = ex.args[0]
            logger.error(f"Database connection failed. SQLSTATE: {sqlstate}", exc_info=True)
            # Provide more user-friendly error messages based on SQLSTATE or exception details
            if "Login failed" in str(ex) or sqlstate == '28000':
                 raise ConnectionError("Database connection failed: Authentication error. Check credentials or Trusted Connection setup.") from ex
            elif "server not found" in str(ex).lower() or sqlstate == '08001':
                 raise ConnectionError(f"Database connection failed: Server '{self.config.get('server')}' not found or unreachable.") from ex
            elif "database" in str(ex).lower() and "not found" in str(ex).lower():
                 raise ConnectionError(f"Database connection failed: Database '{self.config.get('name')}' not found.") from ex
            else:
                 raise ConnectionError(f"Database connection failed: An unexpected error occurred.") from ex
        except Exception as e:
             logger.error(f"An unexpected error occurred while connecting to the database: {e}", exc_info=True)
             raise ConnectionError(f"An unexpected error occurred while connecting to the database: {e}") from e


    def close_connection(self, conn):
        """
        Closes the provided database connection.
        """
        if conn:
            try:
                conn.close()
                logger.info("Database connection closed.")
            except pyodbc.Error as ex:
                 logger.error("Error closing database connection.", exc_info=True)
            except Exception as e:
                 logger.error(f"An unexpected error occurred while closing connection: {e}", exc_info=True)

# Example usage (Test Block) - useful for verifying connectivity
if __name__ == "__main__":
    # Setup logging before loading config and using the connector
    # In a real app, this would be done once at the very beginning.
    try:
        from utils.logging_config import setup_logging
        # Need a dummy config for logging setup if running this file directly
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' } # Use DEBUG level for test output
        setup_logging(DummyConfigForLoggingTest())
        logger.info("Temporary logging setup for db_connector test.")
    except ImportError:
        print("Could not import utils.logging_config. Logging output might be basic.", file=sys.stderr)
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)


    try:
        logger.info("--- Testing DatabaseConnector ---")
        # Load configuration
        app_config = load_config()

        # Initialize the connector
        db_connector = DatabaseConnector(app_config)

        # Get a connection
        conn = db_connector.get_connection()

        # If connection is successful, you can optionally execute a simple query
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION;") # Simple query to get SQL Server version
            row = cursor.fetchone()
            logger.info(f"Successfully queried database. SQL Server Version: {row[0]}")
            cursor.close()

            # Close the connection
            db_connector.close_connection(conn)

        logger.info("--- DatabaseConnector Test Complete ---")

    except ConnectionError as e:
        logger.error(f"Test failed: Database connection error: {e}")
    except Exception as e:
        logger.error(f"Test failed: An unexpected error occurred: {e}", exc_info=True)