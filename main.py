# main.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
print("DEBUG: Loading main.py - Version Check 13") # Increment main.py version
# --- END ADDITION ---


import uvicorn
import logging
import sys
import os

# Import your utility and integration modules
# Ensure these paths are correct relative to your project root
try:
    from utils.config_loader import load_config, AppConfig
    from utils.logging_config import setup_logging
    from integrations.database.db_connector import DatabaseConnector
    from integrations.database.schema_loader import SchemaLoader
    from integrations.openai.openai_client import OpenAIClient
    # Import SQLSafetyChecker and QueryExecutor classes for initialization and dependency injection
    from integrations.database.sql_safety import SQLSafetyChecker
    from integrations.database.query_executor import QueryExecutor # Needed for dependency in routes

    # --- Import ConversationManager ---
    from core.conversation_manager import ConversationManager

    from builtins import ConnectionError # Built-in exception

    # Get module logger here AFTER imports
    # This logger instance will be used throughout the module
    # It gets configured by setup_logging in the startup event
    logger = logging.getLogger(__name__)
    # Basic config for initial messages before setup_logging runs in startup
    # This ensures messages like "Basic logging configured..." and "Attempting to start..." are shown
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logger.info("Basic logging configured for pre-startup messages.")

except ImportError as e:
    # If core imports fail, print a fatal error and exit immediately
    print(f"FATAL ERROR: Failed to import core application modules: {e}", file=sys.stderr)
    sys.exit(1) # Exit with a non-zero status code

# --- Application State Object ---
# Use a class instance to hold initialized services
class AppState:
    def __init__(self):
        self.app_config: AppConfig | None = None
        self.db_connector: DatabaseConnector | None = None
        self.schema_loader: SchemaLoader | None = None
        self.openai_client: OpenAIClient | None = None
        self.sql_safety_checker_instance: SQLSafetyChecker | None = None
        # --- Add conversation_manager_instance attribute to state object ---
        self.conversation_manager_instance: ConversationManager | None = None


# Create a single instance of the state object at module level
# This instance will be modified in the startup event
app_state = AppState()

# --- Global Conversation Manager Instance (Managed via state object) ---
# --- REMOVE the separate global variable definition ---
# conversation_manager_instance: ConversationManager | None = None # <--- DELETE THIS LINE


# --- FastAPI Application Instance ---
from fastapi import FastAPI, Request, HTTPException, status # Import FastAPI here
app = FastAPI(
    title="AI Database Assistant API",
    description="API for interacting with the database using natural language.",
    version="0.1.0",
    # You can add configuration for docs URLs, etc., here
)

# --- Application Startup Event ---
# This function runs *asynchronously* when uvicorn starts the application
@app.on_event("startup")
async def startup_event():
    """
    Handles application startup: loads config, sets up logging,
    initializes database and OpenAI clients, loads schema, initializes safety checker,
    initializes conversation manager.
    Initializes services into the app_state object.
    """
    # Declare app_state as global since we are modifying its attributes
    global app_state
    # No need to declare 'logger' as global, it's a module-level variable being used.

    # Use the module-level logger directly. This is the first place logger is used.
    # The UnboundLocalError traceback points to this line (or very close to it depending on file version).
    # This happens if there's a later assignment to 'logger' in this function's scope.
    logger.info("Application startup started.") # This line must use the module-level logger

    try:
        # 1. Load Configuration (into state object)
        app_state.app_config = load_config()
        logger.info("Configuration loaded.") # Use the module-level logger

        # 2. Setup Logging (using the loaded config)
        # This configures the root logger and potentially the module loggers.
        # The 'logger' instance obtained at the top will automatically use this configuration.
        setup_logging(app_state.app_config)
        # --- CRITICAL FIX: DELETE THIS LINE! It causes the UnboundLocalError ---
        # This assignment makes 'logger' local to this function, causing the error
        # when the line above it is executed before this assignment.
        # If this line exists in your main.py startup_event function, DELETE IT.
        # logger = logging.getLogger(__name__) # <-- THIS LINE MUST BE DELETED

        logger.info("Logging setup complete.") # This uses the now fully configured module-level logger
        logger.info("Configuration and Logging setup complete.")


        # 3. Initialize Database Connector (into state object)
        try:
            app_state.db_connector = DatabaseConnector(app_state.app_config)
            logger.info("DatabaseConnector initialized successfully.") # Use the module-level logger
        # Catch the built-in ConnectionError directly
        except ConnectionError as e: # Use the built-in ConnectionError
            logger.critical(f"FATAL: Failed to connect to database on startup: {e}") # Use the module-level logger
            app_state.db_connector = None # Assign None to state object attribute
            logger.warning("Database connection failed. Database features will be unavailable.") # Use the module-level logger
        except Exception as e:
             logger.critical(f"FATAL: An unexpected error occurred during DatabaseConnector initialization: {e}", exc_info=True) # Use the module-level logger
             app_state.db_connector = None
             logger.warning("Database connection failed unexpectedly. Database features will be unavailable.") # Use the module-level logger


        # 4. Initialize Schema Loader (into state object)
        if app_state.db_connector: # Check attribute of state object
            try:
                app_state.schema_loader = SchemaLoader(app_state.db_connector)
                logger.info("SchemaLoader initialized successfully.") # Use the module-level logger
                # Optional: Load schema eagerly here if needed (e.g., schema_loader.get_full_schema())
            except Exception as e:
                 logger.error(f"Failed to initialize SchemaLoader: {e}", exc_info=True) # Use the module-level logger
                 app_state.schema_loader = None
                 logger.warning("Schema loading failed. Database query features may not work correctly.") # Use the module-level logger
        else:
             logger.warning("Skipped SchemaLoader initialization because DatabaseConnector failed.") # Use the module-level logger


        # 5. Initialize OpenAI Client (into state object)
        try:
            app_state.openai_client = OpenAIClient(app_state.app_config)
            logger.info("OpenAIClient initialized successfully.") # Use the module-level logger
            if app_state.app_config and not app_state.app_config.openai.get('api_key'):
                 logger.warning("OpenAI API key is missing. OpenAI features will be disabled.") # Use the module-level logger
        except Exception as e:
             logger.error(f"Failed to initialize OpenAIClient: {e}", exc_info=True) # Use the module-level logger
             app_state.openai_client = None
             logger.warning("OpenAIClient initialization failed. OpenAI features will be unavailable.") # Use the module-level logger

        # 6. Initialize SQL Safety Checker (into state object)
        try:
            app_state.sql_safety_checker_instance = SQLSafetyChecker(app_state.app_config)
            logger.info("SQLSafetyChecker initialized successfully.") # Use the module-level logger
        except Exception as e:
             logger.error(f"Failed to initialize SQLSafetyChecker: {e}", exc_info=True) # Use the module-level logger
             app_state.sql_safety_checker_instance = None
             logger.warning("SQL safety checker initialization failed. Cannot safely execute queries.") # Use the module-level logger


        # --- 7. Initialize Conversation Manager (into state object attribute) ---
        try:
            # The ConversationManager __init__ calls _load_history
            app_state.conversation_manager_instance = ConversationManager() # Initialize and assign to state attribute
            logger.info("ConversationManager initialized successfully.") # Use the module-level logger
        except Exception as e:
             logger.error(f"Failed to initialize ConversationManager: {e}", exc_info=True) # Use the module-level logger
             app_state.conversation_manager_instance = None # Assign None to state attribute on failure
             logger.warning("Conversation manager initialization failed. Conversation history will not work.")


        logger.info("Application startup finished.") # Use the module-level logger

    except Exception as e:
        # This catches errors that happen *before* setup_logging, or unhandled errors after.
        # Use the module-level logger if configured, otherwise fallback print.
        if logger.handlers: # Check if logging is configured with handlers
            logger.critical(f"An unhandled critical error occurred during application startup: {e}", exc_info=True)
        else:
            # Fallback print if logging setup failed
            print(f"FATAL ERROR: An unhandled critical error occurred during application startup before logging was configured: {e}", file=sys.stderr)

        # Decide how to handle this critical startup failure based on deployment
        # Raising an exception here will cause uvicorn to report startup failure
        raise RuntimeError(f"Application startup failed: {e}") from e


# --- Application Shutdown Event ---
@app.on_event("shutdown")
async def shutdown_event():
    """
    Handles application shutdown: cleans up resources, saves history.
    """
    logger.info("Application shutdown started.") # Use the module-level logger
    # --- Save history on shutdown ---
    # Access the manager instance from the state object
    if app_state.conversation_manager_instance:
        try:
            app_state.conversation_manager_instance.save_history()
            logger.info("Conversation history saved on shutdown.")
        except Exception as e:
             logger.error(f"Failed to save conversation history on shutdown: {e}", exc_info=True)
             # Decide if save failure should prevent shutdown or just log

    logger.info("Application shutdown finished.")


# --- Include API Routes ---
# Import the router from app/api_routes.py and include it
# This import must happen AFTER the 'app_state' object is defined in main.py
# to avoid circular dependency issues if api_routes.py tries to access globals during import.
from app.api_routes import router as api_router
app.include_router(api_router) # Include the router in the app instance


# --- Health Check Endpoint ---
# The health check stays here, accessing state via app_state attributes
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Basic health check endpoint.
    Reports status of core dependencies (Database, OpenAI, SQL Safety Checker, Conversation Manager).
    Returns 200 OK if healthy or degraded, 503 Service Unavailable if critical dependencies failed.
    """
    # Access state via the module-level app_state object
    health_status = "healthy"
    details = {}

    # Check Database Connector status from state object
    if app_state.db_connector is None:
         health_status = "unhealthy" # Consider DB connection critical
         details["database"] = "disconnected"

    # Check Schema Loader status from state object
    if app_state.schema_loader is None and app_state.db_connector is not None: # Schema loader failed despite DB connection
        health_status = "degraded" if health_status == "healthy" else "unhealthy"
        details["schema_loader"] = "failed_init"

    # Check OpenAI Client status from state object
    if app_state.openai_client is None or not (app_state.app_config and app_state.app_config.openai.get('api_key')):
         health_status = "degraded" if health_status == "healthy" else "unhealthy"
         details["openai"] = "unavailable"

    # Check SQL Safety Checker status from state object
    if app_state.sql_safety_checker_instance is None: # Safety checker is critical for query execution
         health_status = "unhealthy" # Critical dependency failure
         details["sql_safety_checker"] = "failed_init"

    # Check ConversationManager status from state object
    if app_state.conversation_manager_instance is None: # Check attribute on state object
         health_status = "unhealthy" # Consider critical if history required
         details["conversation_manager"] = "failed_init"


    if health_status == "healthy":
        logger.info("Health check status: healthy")
    elif health_status == "degraded":
        logger.warning(f"Health check status: degraded - {details}")
    else: # unhealthy
        logger.error(f"Health check status: unhealthy - {details}")
        # Uncomment below to return 503 for unhealthy
        # raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=details)


    return {"status": health_status, "details": details}


# --- Main block to run the FastAPI app ---
# This block is executed when the script is run directly (e.g., python main.py)
# Uvicorn is typically started by calling this block.
if __name__ == "__main__":
    # Read server binding configuration from environment variables
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    uvicorn_log_level = os.environ.get("UVICORN_LOG_LEVEL", "info").lower()

    logger.info(f"Attempting to start FastAPI application with uvicorn on http://{host}:{port}")

    try:
        uvicorn.run(app, host=host, port=port, log_level=uvicorn_log_level)
    except Exception as e:
        print(f"FATAL ERROR: An error occurred during uvicorn execution: {e}", file=sys.stderr)
        sys.exit(1)