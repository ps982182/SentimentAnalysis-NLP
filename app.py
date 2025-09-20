import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Download NLTK data quietly
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# Load your lightweight custom models (uses <50MB instead of 500MB+)
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Custom models loaded successfully")
except FileNotFoundError as e:
    print(f"Model files not found: {e}")
    model = None
    vectorizer = None

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API is running!", 200

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if text == '':
            return jsonify({'error': 'Empty input text'}), 400
        
        clean_text = preprocess(text)
        vect = vectorizer.transform([clean_text])
        pred = model.predict(vect)[0]
        
        # Get confidence score
        try:
            confidence = model.predict_proba(vect)[0].max()
        except:
            confidence = 0.9  # fallback
        
        sentiment = 'positive' if pred == 1 else 'negative'
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': round(float(confidence), 4)
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# CRITICAL: Render requires this exact configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Default to 10000 as per Render docs
    app.run(host='0.0.0.0', port=port, debug=False)
