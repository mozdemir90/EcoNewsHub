from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models
models = {}
vectorizers = {}
w2v_model = None
glove_vectors = None

def load_models():
    """Load all trained models"""
    print("🔄 Loading models...")
    
    # Load TF-IDF vectorizer
    try:
        vectorizers['tfidf'] = joblib.load('models/tfidf_vectorizer.pkl')
        print("✅ TF-IDF vectorizer loaded")
    except:
        print("❌ TF-IDF vectorizer not found")
    
    # Load supervised learning models
    model_types = ['rf', 'svm', 'nb', 'ada', 'ann']
    assets = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
    
    for asset in assets:
        models[asset] = {}
        for model_type in model_types:
            try:
                model_path = f'models/{asset}_{model_type}_model.pkl'
                if os.path.exists(model_path):
                    models[asset][model_type] = joblib.load(model_path)
                    print(f"✅ {asset}_{model_type} loaded")
            except Exception as e:
                print(f"❌ {asset}_{model_type}: {e}")
    
    # Load Word2Vec model
    try:
        w2v_model = Word2Vec.load('models/word2vec_model.model')
        print("✅ Word2Vec model loaded")
    except:
        print("❌ Word2Vec model not found")
    
    # Load GloVe vectors
    try:
        glove_vectors = KeyedVectors.load_word2vec_format('data/glove.6B.100d.txt', binary=False, no_header=True)
        print("✅ GloVe vectors loaded")
    except:
        print("❌ GloVe vectors not found")
    
    print("🎉 Model loading completed!")

def preprocess_text(text):
    """Preprocess text for prediction"""
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    
    return text

def get_tfidf_prediction(text, asset):
    """Get TF-IDF based prediction"""
    try:
        if 'tfidf' not in vectorizers:
            return None
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Transform text
        X = vectorizers['tfidf'].transform([processed_text])
        
        # Get predictions from all models
        predictions = {}
        for model_type in ['rf', 'svm', 'nb', 'ada', 'ann']:
            if asset in models and model_type in models[asset]:
                pred = models[asset][model_type].predict(X)[0]
                pred = min(5, max(1, round(pred)))
                predictions[model_type] = pred
        
        return predictions
    except Exception as e:
        print(f"TF-IDF prediction error: {e}")
        return None

def get_word2vec_prediction(text, asset):
    """Get Word2Vec based prediction"""
    try:
        if w2v_model is None:
            return None
        
        # Preprocess text
        processed_text = preprocess_text(text)
        tokens = processed_text.split()
        
        # Get sentence vector
        vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if not vectors:
            return None
        
        sentence_vector = np.mean(vectors, axis=0).reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        for model_type in ['rf', 'svm', 'ada', 'ann']:
            model_key = f'{asset}_{model_type}_w2v'
            if asset in models and model_type in models[asset]:
                pred = models[asset][model_type].predict(sentence_vector)[0]
                pred = min(5, max(1, round(pred)))
                predictions[model_type] = pred
        
        return predictions
    except Exception as e:
        print(f"Word2Vec prediction error: {e}")
        return None

def get_glove_prediction(text, asset):
    """Get GloVe based prediction"""
    try:
        if glove_vectors is None:
            return None
        
        # Preprocess text
        processed_text = preprocess_text(text)
        tokens = processed_text.split()
        
        # Get sentence vector
        vectors = [glove_vectors[word] for word in tokens if word in glove_vectors]
        if not vectors:
            return None
        
        sentence_vector = np.mean(vectors, axis=0).reshape(1, -1)
        
        # Get predictions from all models
        predictions = {}
        for model_type in ['rf', 'svm', 'ada', 'ann']:
            model_key = f'{asset}_{model_type}_glove'
            if asset in models and model_type in models[asset]:
                pred = models[asset][model_type].predict(sentence_vector)[0]
                pred = min(5, max(1, round(pred)))
                predictions[model_type] = pred
        
        return predictions
    except Exception as e:
        print(f"GloVe prediction error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Financial News Sentiment Analysis API is running',
        'models_loaded': len(models) > 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400
        
        text = data['text']
        method = data.get('method', 'glove')  # Default to GloVe
        asset = data.get('asset', 'all')  # Default to all assets
        
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Define assets
        assets = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        asset_names = {
            'dolar_skor': 'USD',
            'altin_skor': 'Gold',
            'borsa_skor': 'Stock Market',
            'bitcoin_skor': 'Bitcoin'
        }
        
        results = {}
        
        # Get predictions based on method
        if method.lower() == 'tfidf':
            for asset_key in assets:
                if asset == 'all' or asset_key == asset:
                    pred = get_tfidf_prediction(text, asset_key)
                    if pred:
                        results[asset_names[asset_key]] = pred
        
        elif method.lower() == 'word2vec':
            for asset_key in assets:
                if asset == 'all' or asset_key == asset:
                    pred = get_word2vec_prediction(text, asset_key)
                    if pred:
                        results[asset_names[asset_key]] = pred
        
        elif method.lower() == 'glove':
            for asset_key in assets:
                if asset == 'all' or asset_key == asset:
                    pred = get_glove_prediction(text, asset_key)
                    if pred:
                        results[asset_names[asset_key]] = pred
        
        else:
            return jsonify({'error': 'Invalid method. Use: tfidf, word2vec, or glove'}), 400
        
        if not results:
            return jsonify({'error': 'No predictions available. Check if models are loaded.'}), 500
        
        return jsonify({
            'success': True,
            'text': text,
            'method': method,
            'predictions': results,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        method = data.get('method', 'glove')
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Texts must be a non-empty array'}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                # Get prediction for each text
                pred_data = {'text': text, 'method': method, 'asset': 'all'}
                response = predict()
                
                if response.status_code == 200:
                    result = response.get_json()
                    results.append({
                        'index': i,
                        'text': text,
                        'predictions': result['predictions']
                    })
                else:
                    results.append({
                        'index': i,
                        'text': text,
                        'error': 'Prediction failed'
                    })
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'text': text,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'method': method,
            'total_texts': len(texts),
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models"""
    try:
        status = {
            'tfidf_vectorizer': 'tfidf' in vectorizers,
            'word2vec_model': w2v_model is not None,
            'glove_vectors': glove_vectors is not None,
            'supervised_models': {}
        }
        
        # Check supervised models
        assets = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        model_types = ['rf', 'svm', 'nb', 'ada', 'ann']
        
        for asset in assets:
            status['supervised_models'][asset] = {}
            for model_type in model_types:
                status['supervised_models'][asset][model_type] = (
                    asset in models and model_type in models[asset]
                )
        
        return jsonify({
            'success': True,
            'models_status': status,
            'total_models_loaded': sum(len(models.get(asset, {})) for asset in assets)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get model status: {str(e)}'}), 500

@app.route('/models/reload', methods=['POST'])
def reload_models():
    """Reload all models"""
    try:
        global models, vectorizers, w2v_model, glove_vectors
        models = {}
        vectorizers = {}
        w2v_model = None
        glove_vectors = None
        
        load_models()
        
        return jsonify({
            'success': True,
            'message': 'Models reloaded successfully',
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to reload models: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the API
    app.run(debug=True, host='0.0.0.0', port=5000)
