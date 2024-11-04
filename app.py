import numpy as np
import spacy
import time
from gensim.models import Word2Vec
import streamlit as st
import logging
from tensorflow.keras.models import load_model
from database import insert_user_details, close_connection, get_user_details

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


st.markdown("<h1 style='text-align: center; color: #123a7a; font-family: Arial Black;'>iNEURON INTERNSHIP </h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #041633; font-family: Arial Black;'>CONSUMER COMPLAINT ANAYLYSIS </h1>", unsafe_allow_html=True)


if st.button('START PREDICTION'):
    pass


# Streamlit sidebar
st.sidebar.markdown("<h1 style='text-align: center; color:  #8e0ced ;'>iNEURON INTERNSHIP</h1>", unsafe_allow_html=True)


# Load models
def load_models():
    logging.info("Loading models...")
    lstm_model = load_model('TRAINED_lstm_model.h5')
    word2vec_model = Word2Vec.load("word2vec_model.bin")
    logging.info("Models loaded successfully.")
    return lstm_model, word2vec_model

# Get user inputs and concatenate them
def get_user_inputs():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        product_name = st.text_area('PRODUCT NAME ')
    with col2:
        issue = st.text_area('ISSUE')
    with col3:
        company_public_response = st.text_area('COMPANY PUBLIC RESPONSE')
    with col4:
        company_name = st.text_area('COMPANY NAME')

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        tags = st.text_area('TAGS')
    with col6:
        submission_location = st.text_area('SUBMITTED THROUGH')
    with col7:
        company_response = st.text_area("RESPONSE FROM COMPANY ")
    with col8:
        timely_response = st.text_area('RESPONDED ON TIME (YES/NO)')

    user_inputs = [product_name, issue, company_public_response, company_name, tags, submission_location, company_response, timely_response]
    example_text = ', '.join(user_inputs)
    return example_text

# Creating a function that returns the vectorized text of the input given by user
def process_example_text(example_text, word2vec_model):

    nlp = spacy.blank("en")
    doc = nlp(example_text)
    cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
    example_vectors = [word2vec_model.wv[token] for token in cleaned_tokens if token in word2vec_model.wv]
    if not example_vectors:
        example_vectors.append(np.zeros(word2vec_model.vector_size))

    example_vector = np.mean(example_vectors, axis=0)
    return example_vector

# Make final prediction
def make_prediction(lstm_model, example_vector):
    final_vec = example_vector.reshape((1, example_vector.shape[0], 1))
    prediction = lstm_model.predict(final_vec)[0][0]
    return "THE PROBABILTY OF YOU BEING  DISPUTED IS HIGH " if prediction > 0.5 else "ISSUE UNLIKELY TO BE DISPUTED"


# Main code
def main():
    logging.info("Starting the application...")
    lstm_model, word2vec_model = load_models()

    show_details = st.checkbox(" START PREDICTION ")

    if show_details:
        example_text = get_user_inputs()
        input_values = example_text.split(',')
        all_fields_filled = all(input_val.strip() != '' for input_val in input_values)

        if show_details:
            if not all_fields_filled:
                st.warning("PLEASE FILL IN ALL FIELDS BEFORE MAKING A PREDICTION.")   
            elif st.button('DISPLAY PREDICTION'):
                example_vector = process_example_text(example_text, word2vec_model)
                prediction = make_prediction(lstm_model, example_vector)
                with st.spinner('Wait for it...'):
                    time.sleep(2)
                st.markdown(f"<h2 style='text-align: center; color: #522a70;'>Prediction: {prediction}</h2>", unsafe_allow_html=True)
                logging.info(f"PREDICTION ACTION : {prediction}")

                # Logging user details
                logging.info(f"Saving User Details")
if __name__ == '__main__':
    main()
