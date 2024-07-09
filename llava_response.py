import os
import json
import streamlit as st

from llama_index.core.schema import ImageDocument
from llama_index.readers.file.image import ImageReader


def generate_llava_response(prompt_input, mm_model, image_data=None, temperature=0.7, upload_path=None, db_manager=None):

    image_parser = ImageReader(keep_image=True, parse_text=True)
    string_dialogue = """System: You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.
                        Some rules to follow:
                            1. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
                            2. You must focus on OCR task about receipts.
                            3. Answer question from <User> with clarification and high precision, using the provided <Image> and OCR Text on below.
                            4. Information which you create must be listed line by line.
                            5. List of purchased item have structure: <item> : <price>\n.
                            """

    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            if dict_message.get("type") == "image":
                string_dialogue += "User: <Image>\n\n"
            else:
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n" 

    image_documents = []
    ocr_text = ""
    if image_data:
        image_doc = image_parser.load_data(file=upload_path)
        ocr_text = image_doc[0].text if image_doc else ""
        string_dialogue += f"OCR Text: {ocr_text}\n\n"
        image_documents.append(ImageDocument(image_data=image_data))

    # Retrieve similar receipts
    # similar_receipts = db_manager.query(prompt_input)
    # if similar_receipts:
    #     string_dialogue += f"Related Receipt: {similar_receipts}\n\n"

    # string_dialogue += """Refine_prompt: 
    #                     1. If don't have <image> and ocr text, focus on retrieval infomation from <related receipt>"
    #                     2. infomation from <Related Receipt> is used for summarize history,
    #                     retrieval infomation... you must focus on image and ocr text info. 
    #                     """

    response = mm_model.complete(prompt=string_dialogue, image_documents=image_documents)
        
    return response.text, ocr_text
