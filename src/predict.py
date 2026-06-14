import pickle
import os
from src.preprocessing import preprocess

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'emotion_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.pkl')

# Load model, vectorizer, labels
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)

with open(LABELS_PATH, 'rb') as f:
    labels = pickle.load(f)

def predict_emotion(text):
    cleaned = preprocess(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = round(max(probabilities) * 100, 2)
    return prediction, confidence, dict(zip(labels, probabilities))