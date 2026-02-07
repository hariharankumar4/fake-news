import streamlit as st
import joblib
import pandas as pd
import re
import string
import numpy as np
from scipy.sparse import hstack

# 1. Define the text cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Load the saved model and preprocessors
model = joblib.load('model_refined.joblib')
tfidf = joblib.load('tfidf_vectorizer.joblib')
scaler = joblib.load('scaler.joblib')

# 3. Streamlit UI design
st.title("Fake News Detector")
st.markdown("Enter the news title and content below to check if it's real or fake.")

title_input = st.text_input("News Title")
text_input = st.text_area("News Content", height=200)

if st.button("Predict"):
    if title_input.strip() == "" or text_input.strip() == "":
        st.warning("Please provide both a title and the news content.")
    else:
        # Preprocessing
        clean_title = clean_text(title_input)
        clean_content = clean_text(text_input)

        # Feature Engineering: Lengths
        title_char_len = len(clean_title)
        text_char_len = len(clean_content)
        title_word_count = len(clean_title.split())
        text_word_count = len(clean_content.split())

        # Vectorization (TF-IDF on text)
        X_tfidf = tfidf.transform([clean_content])

        # Scaling numerical features
        num_features = np.array([[title_char_len, text_char_len, title_word_count, text_word_count]])
        X_num_scaled = scaler.transform(num_features)

        # Combine features
        X_combined = hstack([X_tfidf, X_num_scaled])

        # Prediction
        prediction = model.predict(X_combined)
        result = "Real News" if prediction[0] == 1 else "Fake News"

        # Display result
        if result == "Real News":
            st.success(f"The prediction is: **{result}**")
        else:
            st.error(f"The prediction is: **{result}**")
