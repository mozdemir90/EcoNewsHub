#!/usr/bin/env python3
"""
Model dosyalarını Google Drive'dan indirme scripti
Deployment sırasında çalıştırılabilir
"""

import os
import requests
import zipfile
from pathlib import Path

def download_from_google_drive(file_id, destination):
    """Google Drive'dan dosya indir"""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

    save_response_content(response, destination)

def download_models():
    """Model dosyalarını indir ve çıkar"""
    
    # Model dizinlerini oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/deep_learning", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Google Drive dosya ID'leri (örnek - gerçek ID'leri kullanın)
    model_files = {
        # GloVe embeddings
        "data/glove.6B.100d.txt": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        
        # TF-IDF models
        "models/tfidf_vectorizer.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/dolar_skor_rf_model.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/altin_skor_rf_model.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/borsa_skor_rf_model.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/bitcoin_skor_rf_model.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        
        # Deep Learning models
        "models/deep_learning/tokenizer.pkl": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/deep_learning/cnn_model.h5": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/deep_learning/lstm_model.h5": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "models/deep_learning/cnn_lstm_model.h5": "1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    }
    
    print("📥 Model dosyaları Google Drive'dan indiriliyor...")
    
    for filepath, file_id in model_files.items():
        try:
            print(f"İndiriliyor: {filepath}")
            download_from_google_drive(file_id, filepath)
            print(f"✅ {filepath} indirildi")
            
        except Exception as e:
            print(f"❌ {filepath} indirilemedi: {e}")
    
    print("🎉 Model indirme tamamlandı!")

if __name__ == "__main__":
    download_models()
