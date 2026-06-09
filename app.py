
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (if not already downloaded)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    WordNetLemmatizer().lemmatize('test')
except LookupError:
    nltk.download('wordnet')

# Load the best model and vectorizer
best_model = joblib.load('best_sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Preprocessing Functions (same as in the notebook)
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces
    return text

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    words = text.split()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmas)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

# Streamlit App Layout
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")
st.title("Sentiment Analysis for Movie Reviews")
st.markdown("Enter a movie review below to get a sentiment prediction (Positive/Negative).")

# Text input from user
user_input = st.text_area("Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_text(user_input)
        
        # Vectorize the preprocessed text
        vectorized_input = vectorizer.transform([processed_input])
        
        # Make prediction
        prediction = best_model.predict(vectorized_input)
        
        # Display result
        st.subheader("Prediction:")
        if prediction[0] == 'positive':
            st.success("This review is likely **Positive!** 👍")
        else:
            st.error("This review is likely **Negative!** 👎")
        
        st.markdown("--- App Information ---")
        st.write(f"**Processed text:** {processed_input}")
    else:
        st.warning("Please enter some text to predict sentiment.")

st.caption("Built with Streamlit and scikit-learn.")
