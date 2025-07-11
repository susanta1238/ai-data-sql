# app/api_routes.py

# --- DIAGNOSTIC STEP: Add a unique print statement AT THE VERY TOP ---
print("DEBUG: Loading app/api_routes.py - Version Check 19") # Increment version for clarity
# --- END ADDITION ---


import logging
import pandas as pd

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

# Import necessary components (types)
from integrations.database.db_connector import DatabaseConnector
from integrations.database.schema_loader import SchemaLoader
from integrations.openai.openai_client import OpenAIClient
from integrations.database.sql_safety import SQLSafetyChecker
from integrations.database.query_executor import QueryExecutor

# --- Import ConversationManager type ---
from core.conversation_manager import ConversationManager

from core.processing_pipeline import ConversationPipeline

# --- IMPORT THE STATE OBJECT FROM main.py ---
# The ConversationManager instance is now an attribute of app_state
from main import app_state # <-- Import the state object


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["chat"],
)

# --- Dependency Injectors (Getter Functions) ---
# (These access attributes of the app_state object)

def get_db_connector() -> DatabaseConnector:
    print("DEBUG: Inside get_db_connector function entry (print).")
    logger.debug("DEBUG: Inside get_db_connector function entry (logger).")
    logger.debug(f"Inside get_db_connector. Value of app_state.db_connector: {app_state.db_connector} (Type: {type(app_state.db_connector)})")
    if app_state.db_connector is None:
        logger.error("Dependency requested DatabaseConnector, but it is not available in app_state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not available. Check API logs for startup errors."
        )
    return app_state.db_connector

def get_schema_loader() -> SchemaLoader:
    if app_state.schema_loader is None:
        logger.error("Dependency requested SchemaLoader, but it is not available in app_state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Schema loading service is unavailable. Check API logs for startup errors."
        )
    return app_state.schema_loader

def get_openai_client() -> OpenAIClient:
    if app_state.openai_client is None or not (app_state.app_config and app_state.app_config.openai.get('api_key')):
        logger.error("Dependency requested OpenAIClient, but it is not available in app_state or API key missing.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI (OpenAI) service is not available. Check API logs for startup errors or missing API key."
        )
    return app_state.openai_client

def get_sql_safety_checker() -> SQLSafetyChecker:
    if app_state.sql_safety_checker_instance is None:
         logger.error("Dependency requested SQLSafetyChecker, but it is not available in app_state.")
         raise HTTPException(
              status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
              detail="SQL safety checker service is unavailable. Cannot process data queries safely."
         )
    return app_state.sql_safety_checker_instance

# --- Update dependency for ConversationManager ---
def get_conversation_manager() -> ConversationManager:
   """Dependency that provides the initialized ConversationManager from app_state."""
   # Check the attribute on the app_state object
   if app_state.conversation_manager_instance is None: # <-- This is the check that was failing
       logger.error("Dependency requested ConversationManager, but it is not available in app_state.")
       raise HTTPException(
           status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
           detail="Conversation management service is unavailable."
       )
   return app_state.conversation_manager_instance # Return the instance from the state object


# --- Pydantic Models for Request and Response Bodies ---
# (These remain the same)
class ChatRequest(BaseModel):
    user_message: str
    conversation_id: str | None = None # conversation_id is now crucial for history

class ChatResponse(BaseModel):
    status: str = "success"
    response_type: str
    data: list[dict] | str | None = None
    message: str | None = None
    # Optional: Include generated SQL in development/staging for transparency
    # generated_sql: str | None = None
    conversation_id: str # Return the conversation ID so client can use it for next turn


# --- API Endpoint for Chat ---

@router.post("/chat", response_model=ChatResponse) # Specify the response model
async def chat_endpoint(
    request_body: ChatRequest, # Use the Pydantic model for request body

    # Use dependency injection to get initialized services from the state object
    db: DatabaseConnector = Depends(get_db_connector),
    schema: SchemaLoader = Depends(get_schema_loader),
    openai: OpenAIClient = Depends(get_openai_client), # Use the real dependency
    sql_checker: SQLSafetyChecker = Depends(get_sql_safety_checker), # Use the real dependency
    # --- Add ConversationManager dependency ---
    conv_manager: ConversationManager = Depends(get_conversation_manager), # Get manager instance via dependency

):
    """
    Processes user natural language input. Handles conversation history.
    Classifies intent (data query or conversation),
    generates/validates/executes SQL for data queries,
    or generates text responses for conversational queries.
    If no conversation_id is provided, a new one is created.
    """
    user_message = request_body.user_message
    conversation_id = request_body.conversation_id

    # If no conversation_id is provided by the client, create a new one using the manager
    if conversation_id is None:
        conversation_id = conv_manager.create_conversation() # Use the manager instance
        logger.info(f"No conversation_id provided, created new one: {conversation_id}")
    else:
         logger.info(f"Using existing conversation_id: {conversation_id}")
         # Optional: Check if the provided conversation_id exists in the manager
         # history = conv_manager.get_history(conversation_id) # Getting history later anyway


    logger.info(f"Received message for conversation {conversation_id}: '{user_message}'")

    # --- Instantiate and run the processing pipeline ---
    # Pass the necessary dependencies (service instances obtained via Depends) to the pipeline constructor.
    try:
        pipeline = ConversationPipeline(
            app_config=app_state.app_config, # Access the config from the state object
            db_connector=db, # Get from dependency
            schema_loader=schema, # Get from dependency
            openai_client=openai, # Get from dependency
            sql_safety_checker=sql_checker, # Get from dependency
            # --- Pass ConversationManager instance ---
            conv_manager=conv_manager, # Pass manager instance from dependency
        )

        # Process the message asynchronously using the pipeline
        # The pipeline will handle retrieving history using its conv_manager instance
        # and adding the turn to history after processing.
        # Pass the conversation ID to the pipeline
        response = await pipeline.process_message(user_message, conversation_id)

        # --- The pipeline is now responsible for adding the turn to history ---
        # The add_turn call has been moved inside process_message.


        # --- Format Response Based on Pipeline Output ---
        response_type = response.get('type')
        response_content = response.get('content')
        generated_sql_output = response.get('generated_sql') # Get generated SQL

        if response_type == 'data':
            formatted_data_output = response_content
            if isinstance(response_content, pd.DataFrame):
                formatted_data_output = response_content.to_dict(orient='records')

            return ChatResponse(
                status="success",
                response_type="data",
                data=formatted_data_output,
                # generated_sql=generated_sql_output # Include generated SQL in response
                conversation_id=conversation_id # Return the ID
            )

        elif response_type == 'text':
             return ChatResponse(
                 status="success",
                 response_type="text",
                 message=response_content,
                 conversation_id=conversation_id # Return the ID
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

    except ValueError as e:
         logger.error(f"Caught ValueError from pipeline: {e}", exc_info=True)
         raise HTTPException(
              status_code=status.HTTP_400_BAD_REQUEST,
              detail=f"Request processing failed: {e}"
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

# --- Endpoint to Create New Conversation (Optional but Recommended) ---
# This allows the client to explicitly request a fresh conversation context.
# @router.post("/conversations", response_model=dict) # Define response model (e.g., {"conversation_id": "..."})
# async def create_new_conversation_endpoint(conv_manager: ConversationManager = Depends(get_conversation_manager)):
#     """Creates a new conversation and returns its ID."""
#     conversation_id = conv_manager.create_conversation()
#     logger.info(f"API endpoint created new conversation: {conversation_id}")
#     return {"conversation_id": conversation_id}