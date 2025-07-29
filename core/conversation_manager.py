# core/conversation_manager.py
import logging
import uuid
import json 
import os   
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# --- Configuration for persistence file ---
# Use an environment variable for the file path, with a default
HISTORY_FILE_PATH = os.environ.get("HISTORY_FILE_PATH", "data/conversation_history.json")
MAX_HISTORY_LENGTH = 10 # Store the last 10 turns (user + assistant messages = 20 messages)

# In-memory storage for conversation history (will be loaded/saved to file)
_conversation_store: Dict[str, List[Dict[str, str]]] = {}

class ConversationManager:
    """
    Manages conversation history storage and retrieval.
    (File-based persistence for demonstration - NOT PRODUCTION READY)
    """
    def __init__(self):
        logger.info(f"ConversationManager initialized. Persistence file: {HISTORY_FILE_PATH}")
        # Load history from file when initialized
        self._load_history()

    def _load_history(self):
        """
        Loads conversation history from the JSON file.
        """
        global _conversation_store
        if os.path.exists(HISTORY_FILE_PATH):
            try:
                with open(HISTORY_FILE_PATH, 'r') as f:
                    _conversation_store = json.load(f)
                logger.info(f"Loaded history from {HISTORY_FILE_PATH}. {len(_conversation_store)} conversations loaded.")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from history file {HISTORY_FILE_PATH}: {e}. Starting with empty history.")
                _conversation_store = {} # Start fresh if file is corrupted
            except Exception as e:
                logger.error(f"An unexpected error occurred loading history from {HISTORY_FILE_PATH}: {e}. Starting with empty history.", exc_info=True)
                _conversation_store = {} # Start fresh on other errors
        else:
            logger.info(f"History file {HISTORY_FILE_PATH} not found. Starting with empty history.")
            _conversation_store = {}

    def save_history(self):
        """
        Saves current conversation history to the JSON file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(HISTORY_FILE_PATH), exist_ok=True)
        try:
            with open(HISTORY_FILE_PATH, 'w') as f:
                json.dump(_conversation_store, f, indent=4) # Use indent for readability
            logger.info(f"Saved history to {HISTORY_FILE_PATH}. {len(_conversation_store)} conversations saved.")
        except Exception as e:
            logger.error(f"An unexpected error occurred saving history to {HISTORY_FILE_PATH}: {e}", exc_info=True)

    def create_conversation(self) -> str:
        """
        Creates a new conversation and returns a unique ID.
        Also saves history after creation.
        """
        conversation_id = str(uuid.uuid4())
        _conversation_store[conversation_id] = []
        logger.debug(f"Created new conversation with ID: {conversation_id}")
        # Save history immediately after creating a new conversation
        self.save_history() # Save to persist the new ID
        return conversation_id

    def get_history(self, conversation_id: str, limit: Optional[int] = MAX_HISTORY_LENGTH) -> List[Dict[str, str]]:
        """
        Retrieves conversation history for a given ID.
        ... (method remains the same) ...
        """
        history = _conversation_store.get(conversation_id, [])
        start_index = max(0, len(history) - (limit * 2))
        retrieved_history = history[start_index:]
        # logger.debug(f"Retrieved history for conversation {conversation_id}: {len(retrieved_history)} messages.") # Keep debug for clarity
        return retrieved_history

    def add_turn(self, conversation_id: str, user_message: str, assistant_response: Any):
        """
        Adds a user message and the corresponding assistant response to the history.
        Saves history after adding the turn.
        ... (method remains the same) ...
        """
        if conversation_id not in _conversation_store:
            logger.warning(f"Attempted to add turn to non-existent conversation ID: {conversation_id}. Creating new entry.")
            _conversation_store[conversation_id] = []

        # Format the assistant response for storage - convert DataFrames to string representation
        if isinstance(assistant_response, pd.DataFrame):
            # Store a string representation of the DataFrame (e.g., first few rows)
            assistant_response_str = assistant_response.head().to_string() # Or to_json(), to_markdown() etc.
        elif isinstance(assistant_response, dict):
             # If the response is a dict (like the pipeline return), format the content
             content = assistant_response.get('content')
             if isinstance(content, pd.DataFrame):
                 assistant_response_str = content.head().to_string()
             elif isinstance(content, str): # Handle the "no data" string message
                 assistant_response_str = content
             elif isinstance(content, dict): # If content is already a dict (e.g., schema), convert to string
                  assistant_response_str = str(content) # Basic string conversion
             else:
                  assistant_response_str = str(assistant_response) # Fallback to string conversion
        else:
            assistant_response_str = str(assistant_response) # Convert any other type to string

        # Add the user message and assistant response as a pair of messages
        _conversation_store[conversation_id].append({"role": "user", "content": user_message})
        _conversation_store[conversation_id].append({"role": "assistant", "content": assistant_response_str})

        # Trim history to the maximum length (keep the most recent turns)
        num_messages = len(_conversation_store[conversation_id])
        if num_messages > MAX_HISTORY_LENGTH * 2:
             _conversation_store[conversation_id] = _conversation_store[conversation_id][-(MAX_HISTORY_LENGTH * 2):]
             logger.debug(f"Trimmed history for conversation {conversation_id}. Keeping last {MAX_HISTORY_LENGTH} turns.")

        logger.debug(f"Added turn to conversation {conversation_id}. Total messages: {len(_conversation_store[conversation_id])}")

        # --- Save history after adding a turn ---
        self.save_history()


# Example usage (optional - keep for testing this module directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("--- Testing ConversationManager with File Persistence ---")

    # Ensure a dummy data directory exists for local testing
    os.makedirs("data", exist_ok=True)
    # Use a specific test file name
    test_file = "data/test_history.json"
    # Temporarily set the HISTORY_FILE_PATH for this test block
    original_history_file_path = HISTORY_FILE_PATH
    HISTORY_FILE_PATH = test_file

    # Clean up test file from previous runs
    if os.path.exists(test_file):
        os.remove(test_file)
        logger.info(f"Cleaned up previous test file: {test_file}")


    # Test 1: Create, add turn, save (implicitly via add_turn)
    logger.info("\n--- Test 1: Create and Add Turn ---")
    manager1 = ConversationManager() # Loads from empty/non-existent file
    conv_id1 = manager1.create_conversation() # Saves to file
    manager1.add_turn(conv_id1, "First message.", "First response.") # Saves to file
    history1 = manager1.get_history(conv_id1)
    logger.info(f"History in manager1 after 1 turn: {history1}")


    # Test 2: Create a new manager instance, load history, add another turn
    logger.info("\n--- Test 2: Load and Add Another Turn ---")
    manager2 = ConversationManager() # Loads history from the file saved by manager1
    history_in_manager2 = manager2.get_history(conv_id1)
    logger.info(f"History loaded in manager2 for {conv_id1}: {history_in_manager2}")
    manager2.add_turn(conv_id1, "Second message.", "Second response.") # Saves to file
    history2 = manager2.get_history(conv_id1)
    logger.info(f"History in manager2 after 2 turns: {history2}")


    # Test 3: Load again to confirm save worked
    logger.info("\n--- Test 3: Confirm Save ---")
    manager3 = ConversationManager() # Loads history saved by manager2
    history_in_manager3 = manager3.get_history(conv_id1)
    logger.info(f"History loaded in manager3 for {conv_id1}: {history_in_manager3}")


    # Test history limit (add many turns to one conversation)
    logger.info(f"\n--- Test 4: History Limit ({MAX_HISTORY_LENGTH}) ---")
    conv_id_limit = manager3.create_conversation() # Saves to file
    logger.info(f"Created new conversation for limit test: {conv_id_limit}")
    for i in range(MAX_HISTORY_LENGTH + 2): # Add 2 extra turns
        manager3.add_turn(conv_id_limit, f"User {i+1}", f"Assistant {i+1}") # Saves each time

    history_limited = manager3.get_history(conv_id_limit)
    logger.info(f"History for limit test ({conv_id_limit}) after adding turns: {len(history_limited)} messages.")
    # Check if the number of messages is equal to the limit * 2
    if len(history_limited) == MAX_HISTORY_LENGTH * 2:
         logger.info("History limit correctly enforced.")
    else:
         logger.warning("History limit NOT correctly enforced.")
    # Verify the content of the last turn
    # logger.info(f"Last turn in limited history: {history_limited[-2:]}")


    # Test 5: Non-existent conversation
    logger.info("\n--- Test 5: Non-existent Conversation ---")
    history_nonexistent = manager3.get_history("non-existent-id")
    logger.info(f"History for non-existent ID: {history_nonexistent}")

    logger.info("\n--- ConversationManager Test Complete ---")

    # Restore original file path
    HISTORY_FILE_PATH = original_history_file_path
    # Clean up the test file
    if os.path.exists(test_file):
        os.remove(test_file)
        logger.info(f"Cleaned up test file: {test_file}")