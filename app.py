import streamlit as st
import pandas as pd
import os
from datetime import datetime
import re
from main import check_all_messages, llm  # Import your chatbot logic

# File to store conversation history
CSV_FILE = "conversation_history.csv"

# Initialize conversation history if file doesn't exist
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Timestamp", "User Input", "Bot Response"])
    df.to_csv(CSV_FILE, index=False)

# Load conversation history
def load_history():
    return pd.read_csv(CSV_FILE)

# Save new conversation to history
def save_to_history(user_input, bot_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[timestamp, user_input, bot_response]], columns=["Timestamp", "User Input", "Bot Response"])
    new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)

# Function to process user input
def get_response(user_input):
    # Clean and process input
    split_message = re.split(r'\s+|[,;?!.-]\s*', user_input.lower())
    response, probability = check_all_messages(split_message)

    # Use LLM if predefined responses are not confident
    if probability < 10:
        response = llm.invoke(user_input)
    return response

# Streamlit UI
st.set_page_config(page_title="Chatbot", page_icon="🤖", layout="wide")

# Sidebar menu
menu = st.sidebar.radio("Menu", ["Home", "Conversation History"])

if menu == "Home":
    st.title("🤖 Welcome to the AI Chatbot!")
    st.write("This chatbot uses NLP and ML to provide intelligent responses.")

    user_input = st.text_input("You:", placeholder="Type your message here...")
    if st.button("Send"):
        if user_input.strip():
            # Get bot response
            bot_response = get_response(user_input)

            # Display response
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Bot:** {bot_response}")

            # Save conversation to history
            save_to_history(user_input, bot_response)

elif menu == "Conversation History":
    st.title("📜 Conversation History")
    st.write("View your past conversations with the chatbot.")

    # Load and display history
    history_df = load_history()
    st.dataframe(history_df)

    # Option to clear history
    if st.button("Clear History"):
        os.remove(CSV_FILE)
        st.success("Conversation history cleared!")
