# utils/config_loader.py
import os
import yaml
import logging # Import logging module
import sys     # Import sys for error output in test block
from dotenv import load_dotenv
from urllib.parse import quote_plus # Needed for connection string encoding

# Ensure logging is available for the test block even if not fully configured yet
# This is just for this file's test block, real app startup will call setup_logging properly.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__) # Get a logger for this module

class AppConfig:
    """
    Class to hold application configuration settings.
    Provides convenient access and handles environment variable overrides.
    """
    def __init__(self, config_data):
        # Validate essential top-level sections
        if 'app' not in config_data or 'openai' not in config_data or 'database' not in config_data or 'logging' not in config_data:
             # Use the logger if available, otherwise print
             if logging.getLogger().handlers:
                 logger.error("Missing essential sections (app, openai, database, logging) in config.yaml")
             else:
                 print("Error: Missing essential sections (app, openai, database, logging) in config.yaml", file=sys.stderr)
             raise ValueError("Missing essential sections in config.yaml")

        self.app = config_data.get('app', {})
        self.openai = config_data.get('openai', {})
        self.database = config_data.get('database', {})
        self.logging = config_data.get('logging', {}) # Load logging section

        # --- Load sensitive data and critical settings from environment variables ---
        # Environment variables take precedence over values in config.yaml or .env

        # OpenAI
        # Allow OPENAI_API_KEY env var to override anything in config.yaml
        self.openai['api_key'] = os.environ.get('OPENAI_API_KEY', self.openai.get('api_key'))

        # Database - Load individual components from environment variables
        self.database['server'] = os.environ.get('DB_SERVER', self.database.get('server'))
        self.database['name'] = os.environ.get('DB_NAME', self.database.get('name'))

        # Authentication
        # Check if DB_TRUSTED_CONNECTION environment variable is set and is a "truthy" value
        # Default to false if env var not set, then check config.yaml
        env_trusted_conn = os.environ.get('DB_TRUSTED_CONNECTION')
        if env_trusted_conn is not None:
            self.database['trusted_connection'] = env_trusted_conn.lower() in ('true', '1', 'yes')
        else:
            # Fallback to value in config.yaml if env var is not set
            self.database['trusted_connection'] = self.database.get('trusted_connection', False)

        # If not trusted connection, load user/password from env (preferred) or config.yaml
        if not self.database['trusted_connection']:
            self.database['user'] = os.environ.get('DB_USER', self.database.get('user'))
            self.database['password'] = os.environ.get('DB_PASSWORD', self.database.get('password'))
        else:
            # Ensure user/password are None if using trusted connection
            self.database['user'] = None
            self.database['password'] = None


        # Trust Server Certificate
        # Check if DB_TRUST_CERTIFICATE environment variable is set and is a "truthy" value
        env_trust_cert = os.environ.get('DB_TRUST_CERTIFICATE')
        if env_trust_cert is not None:
             self.database['trust_server_certificate'] = env_trust_cert.lower() in ('true', '1', 'yes')
        else:
            # Fallback to value in config.yaml if env var is not set
            self.database['trust_server_certificate'] = self.database.get('trust_server_certificate', False)

        # Ensure ODBC driver is set (can be specified in env var as well)
        self.database['odbc_driver'] = os.environ.get('DB_ODBC_DRIVER', self.database.get('odbc_driver'))
        if not self.database.get('odbc_driver'):
             logger.warning("DB_ODBC_DRIVER is not set. Using default '{ODBC Driver 17 for SQL Server}'. Ensure correct driver is installed.")
             self.database['odbc_driver'] = "{ODBC Driver 17 for SQL Server}" # Provide a common fallback

    def build_db_connection_string(self):
        """
        Constructs the pyodbc connection string based on loaded database configuration.
        """
        db_conf = self.database

        # Basic validation before building string
        if not db_conf.get('server') or not db_conf.get('name'):
             raise ValueError("Cannot build database connection string: Server or database name is missing.")
        if not db_conf.get('odbc_driver'):
             raise ValueError("Cannot build database connection string: ODBC driver is not specified.")

        params = []
        params.append(f"DRIVER={{{db_conf['odbc_driver'].strip('{}')}}}") # Ensure driver name is correctly formatted with braces
        params.append(f"SERVER={db_conf['server']}")
        params.append(f"DATABASE={db_conf['name']}")

        if db_conf.get('trusted_connection'):
            params.append("Trusted_Connection=yes")
        else:
            if not db_conf.get('user') or not db_conf.get('password'):
                 raise ValueError("Cannot build database connection string: User or password missing for SQL Server Authentication.")
            params.append(f"UID={db_conf['user']}")
            params.append(f"PWD={db_conf['password']}")

        if db_conf.get('trust_server_certificate'):
             params.append("TrustServerCertificate=yes")

        # Join parameters with semicolons and URL-encode
        return ";".join(params)
        # If using SQLAlchemy, you'd wrap this:
        # encoded_params = quote_plus(";".join(params))
        # return f"mssql+pyodbc:///?odbc_connect={encoded_params}"


def load_config():
    """
    Loads configuration from config.yaml and environment variables.
    Environment variables take precedence.
    """
    # Load environment variables from .env file if it exists (for local dev)
    # This should be called early in the application startup
    load_dotenv()
    logger.info("Environment variables loaded from .env (if found).")


    # Get config file path from environment variable, default if not set
    config_file_path = os.environ.get('CONFIG_FILE_PATH', './config/config.yaml')
    logger.info(f"Loading configuration from: {config_file_path}")

    try:
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        if config_data is None: # Handle empty config file
            config_data = {}
            logger.warning(f"Configuration file {config_file_path} is empty.")

    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_file_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config file: {e}")
        raise

    # Wrap config data in our access class (handles env var overrides internally)
    try:
        config = AppConfig(config_data)
    except ValueError as e:
        # AppConfig init already logs missing sections, just re-raise
        raise e
    except Exception as e:
         logger.error(f"An unexpected error occurred while processing configuration data: {e}")
         raise

    # --- Post-loading Validation ---
    # Basic validation for critical sensitive values
    if not config.openai.get('api_key'):
         logger.error("OPENAI_API_KEY is not set in environment or config.")
         # In production, you might raise an exception here
         # raise ValueError("OPENAI_API_KEY is not set.")

    # Validate database connection details (basic check)
    # The build_db_connection_string method also has validation, but this is early feedback
    try:
        config.build_db_connection_string() # Attempt to build to trigger validation
    except ValueError as e:
        logger.error(f"Database connection details are incomplete or invalid: {e}")
        # In production, you might raise an exception here
        # raise e

    logger.info("Configuration loaded and validated.")
    return config

# --- Example usage (Test Block) ---
if __name__ == "__main__":
    # Import setup_logging only when running this file directly
    # This prevents circular import issues when other parts of the app import config_loader
    # before logging is globally configured during main app startup.
    try:
        # Temporarily set up basic logging for this test if not already done
        # In a real app, setup_logging would be called once at startup
        from utils.logging_config import setup_logging
        # We need a dummy config object to call setup_logging for the test if needed
        class DummyConfigForLoggingTest:
             def __init__(self):
                 self.logging = { 'level': 'DEBUG' } # Use DEBUG level for test output
        setup_logging(DummyConfigForLoggingTest())
        logger.info("Temporary logging setup for config_loader test.")
    except ImportError:
        print("Could not import utils.logging_config. Logging output might be basic.", file=sys.stderr)
        # Fallback to basic config if import fails
        if not logging.getLogger().handlers:
             logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)


    try:
        # Attempt to load the actual configuration
        app_config = load_config()

        # Use the logger to print loaded settings
        logger.info("-" * 30)
        logger.info("Configuration Summary:")
        logger.info(f"App Name: {app_config.app.get('name')}")
        logger.info(f"App Port: {app_config.app.get('port')}")
        logger.info(f"OpenAI Model: {app_config.openai.get('model')}")
        # Be cautious printing sensitive keys, even parts
        # logger.info(f"OpenAI API Key (Partial): ...{app_config.openai.get('api_key', '')[-4:]}")
        logger.info(f"DB Server: {app_config.database.get('server')}")
        logger.info(f"DB Name: {app_config.database.get('name')}")
        logger.info(f"DB Trusted Connection: {app_config.database.get('trusted_connection')}")
        logger.info(f"DB Trust Certificate: {app_config.database.get('trust_server_certificate')}")
        logger.info(f"DB ODBC Driver: {app_config.database.get('odbc_driver')}")
        # Avoid printing user/password even if loaded
        # logger.info(f"DB User: {app_config.database.get('user')}")
        logger.info(f"Logging Level: {app_config.logging.get('level')}")

        try:
            conn_str = app_config.build_db_connection_string()
            logger.info(f"Built DB Connection String: {conn_str}")
        except ValueError as e:
             logger.warning(f"Could not build connection string: {e}")


        logger.info("-" * 30)


    except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
        logger.error(f"Fatal Error during config loading test: {e}", exc_info=True) # Log exception details
    except Exception as e:
        logger.error(f"An unexpected fatal error occurred during config loading test: {e}", exc_info=True)