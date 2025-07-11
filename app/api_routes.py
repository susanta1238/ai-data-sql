# app/api_routes.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
print("DEBUG: Loading app/api_routes.py - Version Check 16") # Increment version for clarity
# --- END ADDITION ---


import logging
import pandas as pd # Needed for type hinting DataFrame and formatting results

# Import FastAPI's APIRouter and dependencies
from fastapi import APIRouter, Depends, HTTPException, status

# Import Pydantic for request and response body modeling
from pydantic import BaseModel

# Import necessary components from your core logic and services
# These will be accessed via dependency injection (Depends) using getter functions
# which reference the global instances initialized in main.py.
# We need to import the *types* for type hinting the dependencies.
from integrations.database.db_connector import DatabaseConnector
from integrations.database.schema_loader import SchemaLoader
# Import OpenAIClient class (its generate_text method is async)
from integrations.openai.openai_client import OpenAIClient
from integrations.database.sql_safety import SQLSafetyChecker
# Import QueryExecutor class (RuntimeError is a built-in exception, not imported here)
from integrations.database.query_executor import QueryExecutor # Import QueryExecutor class

# Import the custom ConversationPipeline class from your core logic
from core.processing_pipeline import ConversationPipeline # We'll instantiate this in the endpoint

# --- IMPORT THE STATE OBJECT FROM main.py ---
from main import app_state # <-- Import the state object


# Get logger for this module AFTER the diagnostic print statement
logger = logging.getLogger(__name__)

# Create an API router instance
router = APIRouter(
    prefix="/api", # Adds /api to the start of all routes in this file (e.g., /api/chat)
    tags=["chat"], # Groups these routes under the "chat" tag in the OpenAPI docs (Swagger UI)
)

# --- Dependency Injectors (Getter Functions) ---
# These functions provide access to the global service instances initialized in main.py.
# (These remain the same)
def get_db_connector() -> DatabaseConnector:
    """Dependency that provides the initialized DatabaseConnector from app_state."""
    # --- DEBUG LOGS: Check the value of the db_connector attribute on the state object ---
    # These logs must appear in your terminal when you hit /api/chat if the function is reached.
    print("DEBUG: Inside get_db_connector function entry (print).") # Basic print as fallback
    logger.debug("DEBUG: Inside get_db_connector function entry (logger).")
    # This log shows the value of the db_connector attribute on the app_state object
    logger.debug(f"Inside get_db_connector. Value of app_state.db_connector: {app_state.db_connector} (Type: {type(app_state.db_connector)})")
    # --- END DEBUG LOGS ---

    # Check if the service is available in the state object
    if app_state.db_connector is None:
        logger.error("Dependency requested DatabaseConnector, but it is not available in app_state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not available. Check API logs for startup errors."
        )
    return app_state.db_connector # Return the instance from the state object

def get_schema_loader() -> SchemaLoader:
    """Dependency that provides the initialized SchemaLoader from app_state."""
    # Check if the global instance was successfully initialized
    # If schema_loader is None, it usually means db_connector was also None during startup,
    # so checking db_connector first via its dependency is also implicitly handled if needed.
    if app_state.schema_loader is None:
        logger.error("Dependency requested SchemaLoader, but it is not available in app_state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schema loading service is unavailable. Check API logs for startup errors."
        )
    return app_state.schema_loader

def get_openai_client() -> OpenAIClient:
    """Dependency that provides the initialized OpenAIClient from app_state."""
    # Check the instance and if app_config (also on state) is available for API key check
    if app_state.openai_client is None or not (app_state.app_config and app_state.app_config.openai.get('api_key')):
        logger.error("Dependency requested OpenAIClient, but it is not available in app_state or API key missing.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI (OpenAI) service is not available. Check API logs for startup errors or missing API key."
        )
    return app_state.openai_client

def get_sql_safety_checker() -> SQLSafetyChecker:
    """Dependency that provides the initialized SQLSafetyChecker from app_state."""
    if app_state.sql_safety_checker_instance is None:
         logger.error("Dependency requested SQLSafetyChecker, but it is not available in app_state.")
         raise HTTPException(
              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="SQL safety checker service is unavailable. Cannot process data queries safely."
         )
    return app_state.sql_safety_checker_instance

# TODO: Add dependency for ConversationManager when implemented
# def get_conversation_manager() -> ConversationManager: ...


# --- Pydantic Models for Request and Response Bodies ---
# (These remain the same)
class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str | None = None

class ChatResponse(BaseModel):
    status: str = "success"
    response_type: str
    data: list[dict] | str | None = None
    message: str | None = None
    # generated_sql: str | None = None


# --- API Endpoint for Chat ---

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request_body: ChatRequest,

    # Use dependency injection to get initialized services from the state object
    db: DatabaseConnector = Depends(get_db_connector),
    schema: SchemaLoader = Depends(get_schema_loader),
    openai: OpenAIClient = Depends(get_openai_client), # Use the real dependency
    sql_checker: SQLSafetyChecker = Depends(get_sql_safety_checker), # Use the real dependency
    # TODO: conversation_manager = Depends(get_conversation_manager)

):
    """
    Processes user natural language input.
    Classifies intent (data query or conversation),
    generates/validates/executes SQL for data queries,
    or generates text responses for conversational queries.
    """
    # Access validated data directly from the Pydantic model instance
    user_message = request_body.user_message
    conversation_id = request_body.conversation_id

    logger.info(f"Received message for conversation {conversation_id}: '{user_message}'")

    # --- Instantiate and run the processing pipeline ---
    try:
        pipeline = ConversationPipeline(
            app_config=app_state.app_config, # Access the config from the state object
            db_connector=db, # Get from dependency
            schema_loader=schema, # Get from dependency
            openai_client=openai, # Get from dependency
            sql_safety_checker=sql_checker, # Get from dependency
            # TODO: Pass conversation_manager = Depends(get_conversation_manager)
        )

        response = await pipeline.process_message(user_message, conversation_id)

        # --- Format Response Based on Pipeline Output ---
        response_type = response.get('type')
        response_content = response.get('content')
        generated_sql_output = response.get('generated_sql') # Get generated SQL

        if response_type == 'data':
            # Format the content for the API response.
            # If content is a DataFrame, convert it to a list of dictionaries (JSON serializable).
            # If content is the "no data" string message, keep it as a string in the 'data' field.
            formatted_data_output = response_content # Start with the raw content

            if isinstance(response_content, pd.DataFrame):
                formatted_data_output = response_content.to_dict(orient='records') # Convert DataFrame to list of dicts

            return ChatResponse(
                status="success",
                response_type="data",
                data=formatted_data_output, # This will be list[dict] or string
                # generated_sql=generated_sql_output # Include generated SQL in response
            )

        elif response_type == 'text':
             return ChatResponse(
                 status="success",
                 response_type="text",
                 message=response_content # Put text content in the 'message' field
             )

        else:
            logger.error(f"Pipeline returned unexpected response type: {response_type}. Full response: {response}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error processing the request: Unexpected pipeline response type."
            )


    # --- Exception Handling ---
    except RuntimeError as e:
        logger.error(f"Caught RuntimeError from pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database query execution failed: {e}"
        )

    # Catch ValueError, which the pipeline now raises if LLM says it can't generate SQL or safety check fails
    except ValueError as e:
         logger.error(f"Caught ValueError from pipeline: {e}", exc_info=True)
         # Return 400 for unsafe queries or if LLM explicitly failed to generate
         # You might want different detail messages based on the specific ValueError reason
         # For now, a generic validation error detail:
         raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail=f"Request processing failed: {e}" # Expose error message from pipeline (sanitize in prod!)
         )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Caught unhandled exception in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred during processing."
        )

# TODO: Add other endpoints here if needed (e.g., /schema endpoint)
# This endpoint uses the get_schema_loader dependency
# @router.get("/schema", response_model=dict)
# async def get_db_schema(schema_loader_instance: SchemaLoader = Depends(get_schema_loader)):
#     logger.info("Received request for database schema.")
#     try:
#          full_schema = schema_loader_instance.get_full_schema()
#          # Format schema for API response if needed, or return the dict directly
#          # You might want to use a Pydantic model for the schema response too.
#          return {"status": "success", "schema": full_schema}
#     except Exception as e:
#          logger.error(f"Failed to retrieve schema in /schema endpoint: {e}", exc_info=True)
#          raise HTTPException(
#               status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#               detail="Failed to retrieve database schema."
#          )