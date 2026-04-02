import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Model
# -------------------------------
try:
    model = load_model("nextword_model.h5")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -------------------------------
# Load Tokenizer
# -------------------------------
try:
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# -------------------------------
# Reverse Index
# -------------------------------
try:
    reverse_index = {idx: word for word, idx in tokenizer.word_index.items()}
except Exception as e:
    st.error(f"Error creating reverse index: {e}")
    st.stop()

# -------------------------------
# Max Length (must match training)
# -------------------------------
max_len = 44

st.title("Next Word Prediction with Deep Learning")

# -------------------------------
# 🔥 Sampling Function (Improved)
# -------------------------------
def sample_with_temperature(preds, temperature=0.7, top_k=5):
    preds = np.asarray(preds).astype("float64")

    # Avoid log(0)
    preds = np.log(preds + 1e-10) / temperature

    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    # Top-k filtering
    top_indices = np.argsort(preds)[-top_k:]
    top_probs = preds[top_indices]

    # Normalize again
    top_probs = top_probs / np.sum(top_probs)

    # Random selection
    return np.random.choice(top_indices, p=top_probs)

# -------------------------------
# Text Generation Function
# -------------------------------
def generate_text(seed_text, num_words=10):
    text = seed_text.strip()

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([text])[0]

        # Handle unknown input
        if len(seq) == 0:
            return "⚠️ Input words not in vocabulary"

        padded = pad_sequences([seq], maxlen=max_len, padding="pre")

        preds = model.predict(padded, verbose=0)[0]

        # Use improved sampling
        pos = sample_with_temperature(preds, temperature=0.7, top_k=5)

        next_word = reverse_index.get(pos, "")

        if next_word == "":
            break

        text += " " + next_word

    return text

# -------------------------------
# UI Components
# -------------------------------
seed = st.text_input("Enter starting text:", "Hello")

num_words = st.slider("Number of words to generate", 1, 20, 10)

# -------------------------------
# Generate Button
# -------------------------------
if st.button("Generate"):
    if seed.strip() == "":
        st.warning("Please enter some text")
    else:
        result = generate_text(seed, num_words)
        st.success(result)