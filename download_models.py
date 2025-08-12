#!/usr/bin/env python3
"""
Model dosyalarını otomatik indirme scripti
Deployment sırasında çalıştırılabilir
"""

import os
import requests
import zipfile
from pathlib import Path

def download_models():
    """Model dosyalarını indir ve çıkar"""
    
    # Model dizinlerini oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/deep_learning", exist_ok=True)
    
    # Model dosyalarının URL'leri (örnek)
    model_urls = {
        "tfidf_vectorizer.pkl": "https://example.com/models/tfidf_vectorizer.pkl",
        "dolar_skor_rf_model.pkl": "https://example.com/models/dolar_skor_rf_model.pkl",
        # Diğer model dosyaları...
    }
    
    print("📥 Model dosyaları indiriliyor...")
    
    for filename, url in model_urls.items():
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = f"models/{filename}"
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ {filename} indirildi")
            
        except Exception as e:
            print(f"❌ {filename} indirilemedi: {e}")
    
    print("🎉 Model indirme tamamlandı!")

if __name__ == "__main__":
    download_models()
