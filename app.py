import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from database_manager import DatabaseManager
from llava_response import generate_llava_response
from chat_history import clear_chat_history, initialize_chat, display_chat_messages
from llama_index.multi_modal_llms.ollama import OllamaMultiModal

import os

# Load environment variables
load_dotenv()


# Initialize DatabaseManager
# db_manager = DatabaseManager(index_name="Receipts")


# Set up Streamlit app
st.set_page_config(page_title="🦙💬 Receipts Chatbot 📖")

# Sidebar configuration
with st.sidebar:
    st.title('🦙💬 Receipts Chatbot 📖')
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=2.0, value=1.0, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=64, max_value=4096, value=512, step=8)
    st.button('Clear Chat History', on_click=clear_chat_history)
    # if st.button("Reset Database"):
    #     db_manager.reset_database()
    #     db_manager = DatabaseManager(index_name="Receipts")
    #     st.sidebar.write("Database has been reset.")
    # st.header('Summary')
    # month = st.text_input("Enter month (MM):")        
    # year = st.text_input("Enter year (YYYY):")
    # if st.button("Get Monthly Spending"):
    #     if month and year:
    #         total_spending = db_manager.get_monthly_spending(month, year)
    #         st.sidebar.write(f"Total spending for {month}/{year}: ${total_spending:.2f}")
mm_model = OllamaMultiModal(model="llava:7b-v1.6-mistral-q6_K", temperature=temperature)

# Initialize chat
initialize_chat()

# Display chat messages
display_chat_messages()

# File uploader for images
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Text input for chat
prompt = st.text_input("Type a message:")

# Button to send the message/image
if st.button('Send'):
    if prompt:
        st.session_state.messages.append({"role": "user", "content": uploaded_file, "type": "image"})
        st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate a new response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if uploaded_file:
                    response, ocr_text = generate_llava_response(prompt,mm_model=mm_model, image_data=image, temperature=temperature, upload_path=uploaded_file, db_manager=None)
                else:
                    response, _ = generate_llava_response(prompt,mm_model=mm_model, temperature=temperature, db_manager=None)
                
                # new_document = {
                #     "prompt_input": prompt,
                #     "ocr_text": ocr_text if uploaded_file else None,
                # }
                # db_manager.ingest_data([new_document])
                placeholder = st.empty()
                full_response = response
                placeholder.markdown(full_response)

        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
