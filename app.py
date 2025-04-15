import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained LSTM model
model = load_model('next_word_lstm.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict top N next words
def predict_next_words(model, tokenizer, text, max_sequence_len, top_n=3):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = predicted_probs.argsort()[-top_n:][::-1]  # Top N indices

    word_map = {index: word for word, index in tokenizer.word_index.items()}
    predictions = [(word_map.get(i, '<?>'), predicted_probs[i]) for i in top_indices]

    return predictions

# Streamlit App
st.set_page_config(page_title="Next Word Predictor", layout="centered")

st.title("🔮 Next Word Prediction")
st.markdown("Enter a phrase, and let the LSTM model guess the **next word**!")

input_text = st.text_input("📝 Enter your sentence:", "To be or not to")

if st.button("🚀 Predict"):
    max_sequence_len = model.input_shape[1] + 1
    predictions = predict_next_words(model, tokenizer, input_text, max_sequence_len)

    st.subheader("🔍 Top Predictions:")
    for i, (word, prob) in enumerate(predictions, start=1):
        st.write(f"**{i}. `{word}`**  —  Confidence: `{prob:.4f}`")

    st.markdown("---")
    st.info("Tip: Try entering phrases like `I want to`, `Life is`, or `Dream big and`.")