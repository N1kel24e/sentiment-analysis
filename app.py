import streamlit as st
import joblib
import re
from bs4 import BeautifulSoup

def clean_text (text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    return text

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Sentiment Analysis")
st.write("Enter a movie review and I'll tell you if it's positive or negative!")

review = st.text_area("Share your review: ")

if st.button("Analyze"):
    if len(review.split()) <= 4:
        st.warning("Please enter a longer review")
    else:
        cleaned_review = clean_text(review)
        vectorized = vectorizer.transform([cleaned_review])
        result = model.predict(vectorized)
        if result[0] == 1:
            st.success('positive review')
        else:
            st.error('negative review')

