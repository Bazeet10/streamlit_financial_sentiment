import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Set page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def download_nltk_resources():
    import os
    import nltk
    
    # Create directory
    os.makedirs('nltk_data/tokenizers', exist_ok=True)
    os.makedirs('nltk_data/corpora', exist_ok=True)
    
    # Force download
    nltk.download('punkt', download_dir='nltk_data')
    nltk.download('stopwords', download_dir='nltk_data')
    
    # Set the NLTK data path to find these resources
    nltk.data.path.append('nltk_data')

download_nltk_resources()

# Preprocessing function
@st.cache_data
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Load the model and tokenizer from Hugging Face
@st.cache_resource
def load_distilbert_model():
    try:
        model_path = "Bazeet/streamlit-financial-model"
        model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Predict sentiment
def predict_sentiment(text, model, tokenizer):
    processed_text = preprocess_text(text)
    inputs = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='tf'
    )
    outputs = model(inputs)
    logits = outputs.logits.numpy()
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    prediction = np.argmax(probs)
    sentiment = "Positive" if prediction == 1 else "Negative/Neutral"
    confidence = float(probs[prediction])
    return sentiment, confidence

# Main function
def main():
    st.title("📊 Financial Sentiment Analysis")
    st.markdown("### Powered by DistilBERT")

    model, tokenizer = load_distilbert_model()
    if model is None or tokenizer is None:
        st.error("Failed to load the model from Hugging Face.")
        st.stop()

    # User input
    st.subheader("Enter text for sentiment analysis")
    user_input = st.text_area("Type or paste financial text:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment, confidence = predict_sentiment(user_input, model, tokenizer)
            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")
            st.info(f"Confidence: {confidence:.2%}")
            with st.expander("View Preprocessed Text"):
                st.write(preprocess_text(user_input))

if __name__ == "__main__":
    main()
