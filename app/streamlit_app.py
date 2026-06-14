import streamlit as st
import pickle
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'models', 'emotion_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models', 'labels.pkl'), 'rb') as f:
        labels = pickle.load(f)
    return model, vectorizer, labels

model, vectorizer, labels = load_model()

# Emotion emojis
EMOJIS = {
    'anger':    '😠',
    'joy':      '😄',
    'love':     '❤️',
    'sadness':  '😢',
    'surprise': '😲'
}

# Page config
st.set_page_config(page_title="Emotion Detector", page_icon="🧠", layout="centered")

# Header
st.title("🧠 Tweet Emotion Detector")
st.markdown("Detect the emotion behind any tweet or text using Machine Learning.")
st.divider()

# Input
text_input = st.text_area("Enter your text below:", placeholder="e.g. I can't believe how amazing today was!", height=150)

# Predict button
if st.button("🔍 Detect Emotion", use_container_width=True):
    if text_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Preprocess and predict
        cleaned = preprocess(text_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        confidence = round(max(probabilities) * 100, 2)
        scores = dict(zip(labels, probabilities))

        st.divider()

        # Result
        emoji = EMOJIS.get(prediction, '🤔')
        st.markdown(f"## {emoji} Detected Emotion: `{prediction.upper()}`")
        st.markdown(f"**Confidence:** {confidence}%")
        st.progress(confidence / 100)

        st.divider()

        # Confidence bar chart
        st.markdown("### Confidence Scores per Emotion")
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71' if label == prediction else '#3498db' for label in scores.keys()]
        bars = ax.barh(list(scores.keys()), list(scores.values()), color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Confidence')
        for bar, val in zip(bars, scores.values()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val*100:.1f}%', va='center')
        plt.tight_layout()
        st.pyplot(fig)

st.divider()
st.caption("Built with Scikit-Learn · TF-IDF · Streamlit")