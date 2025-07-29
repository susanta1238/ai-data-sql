# frontend/app.py
import streamlit as st
import requests
import json 
import pandas as pd 

# --- Configuration ---
# URL of your running FastAPI backend
API_URL = "http://127.0.0.1:9010/api/chat"

# --- Streamlit App Setup ---
st.set_page_config(page_title="AI Database Assistant", layout="wide")

st.title("💬 LeadNova AI")
st.caption("Interact with your database using natural language.")

# --- Session State Initialization ---
# Initialize conversation_id and messages list in session state
if 'conversation_id' not in st.session_state:
    st.session_state['conversation_id'] = None # Start with no ID
if 'messages' not in st.session_state:
    # Messages stored as [{"role": "user", "content": "..."}] or [{"role": "assistant", "content": "..."}]
    st.session_state['messages'] = []

# --- Display Conversation History ---
# Iterate through messages in session state and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if content is a DataFrame (from a previous data query response)
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"]) # Display DataFrame nicely
        elif isinstance(message["content"], str):
            st.markdown(message["content"]) # Display text with Markdown support
        elif isinstance(message["content"], dict):
            # Handle other potential content types like formatted schema if you add it to history
             st.json(message["content"]) # Display dict as JSON

# --- Chat Input ---
# Get user input from the chat input box
if prompt := st.chat_input("Enter your message..."):
    # Add user message to history immediately
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Send Message to Backend API ---
    # Prepare the request payload
    payload = {
        "user_message": prompt,
        "conversation_id": st.session_state.conversation_id # Send current ID (can be None initially)
    }

    try:
        # Send POST request to FastAPI endpoint
        response = requests.post(API_URL, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()

            # Extract necessary info from the response
            status = response_data.get("status")
            response_type = response_data.get("response_type")
            data_content = response_data.get("data") # Can be list[dict] or string ("no data")
            message_content = response_data.get("message") # Used for text responses
            returned_conv_id = response_data.get("conversation_id") # Get the ID back from backend

            # Update conversation_id in session state if the backend returned one
            # (Backend creates a new ID if None was sent)
            if returned_conv_id:
                 st.session_state.conversation_id = returned_conv_id


            # Process and display the assistant's response
            with st.chat_message("assistant"):
                if response_type == "data":
                    # If response_type is 'data', the content is in the 'data' field.
                    # It can be a list of dicts (actual data) or a string ("no data").
                    if isinstance(data_content, list):
                         # Convert list of dicts to DataFrame for nice display
                         df = pd.DataFrame(data_content)
                         st.dataframe(df)
                         # Add DataFrame to session state history for display on rerun
                         st.session_state.messages.append({"role": "assistant", "content": df})
                    elif isinstance(data_content, str):
                         # Handle the "no data" message from the executor
                         st.markdown(data_content)
                         st.session_state.messages.append({"role": "assistant", "content": data_content})
                    else:
                         st.error(f"Received unexpected data format: {type(data_content)}")
                         st.json(data_content) # Show raw response if unexpected
                         st.session_state.messages.append({"role": "assistant", "content": f"Error: Received unexpected data format: {type(data_content)}"})

                elif response_type == "text":
                    # If response_type is 'text', content is in the 'message' field.
                    st.markdown(message_content)
                    st.session_state.messages.append({"role": "assistant", "content": message_content}) # Add text to history

                else:
                    st.error(f"Received unexpected response type from API: {response_type}")
                    st.json(response_data) # Show raw response if unexpected
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: Received unexpected response type: {response_type}"})


        elif response.status_code >= 400:
            # Handle API errors (4xx or 5xx)
            error_detail = response.json().get("detail", "Unknown API error")
            st.error(f"API Error {response.status_code}: {error_detail}")
            # Add error message to session state history
            st.session_state.messages.append({"role": "assistant", "content": f"API Error {response.status_code}: {error_detail}"})

        else:
            # Handle other non-error status codes if necessary
            st.warning(f"Received unexpected API status code: {response.status_code}")
            st.json(response.json())
            st.session_state.messages.append({"role": "assistant", "content": f"Unexpected API status code: {response.status_code}"})


    except requests.exceptions.ConnectionError as e:
        st.error(f"Failed to connect to the API at {API_URL}. Please ensure your backend FastAPI application is running.")
        st.exception(e) # Show full exception details in the app
        st.session_state.messages.append({"role": "assistant", "content": f"Connection Error: Failed to connect to the API. Please ensure your backend is running."})

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)
        st.session_state.messages.append({"role": "assistant", "content": f"An unexpected error occurred: {e}"})


# --- Optional: Clear History Button ---
# Add a button in the sidebar to clear the current conversation history
with st.sidebar:
    st.header("Options")
    # Add the conversation ID to the sidebar for reference
    st.markdown(f"**Current Conversation ID:** `{st.session_state.conversation_id}`")
    st.info("This is the ID for the current chat session.")
    st.warning("History is saved to file on backend shutdown, but this sidebar feature only shows the current session.")


    # Button to clear the current session history
    if st.button("Clear Chat History"):
        st.session_state['messages'] = [] # Clear messages in session state
        st.session_state['conversation_id'] = None # Clear conversation ID to start fresh
        st.experimental_rerun() # Rerun the app to clear the display


# --- Instructions for Running ---
