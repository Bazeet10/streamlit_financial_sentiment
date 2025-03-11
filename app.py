import streamlit as st
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Set page configuration
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Simple preprocessing function that doesn't rely on NLTK
@st.cache_data
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple split by whitespace - no need for NLTK tokenizer
    tokens = text.split()
    
    # Basic English stopwords list
    basic_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                       'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                       'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                       'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                       'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
                       'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                       'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                       'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                       'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                       'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                       'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                       'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                       'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                       'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                       'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
                       's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
    
    # Filter stopwords
    tokens = [token for token in tokens if token not in basic_stopwords]
    
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
    
    # Add an info message about the preprocessing
    st.info("Note: This app uses a custom preprocessing pipeline to analyze financial text sentiment.")
    
    model, tokenizer = load_distilbert_model()
    if model is None or tokenizer is None:
        st.error("Failed to load the model from Hugging Face.")
        st.stop()
    
    # User input
    st.subheader("Enter text for sentiment analysis")
    user_input = st.text_area("Type or paste financial text:", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input:
            with st.spinner("Analyzing sentiment..."):
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
