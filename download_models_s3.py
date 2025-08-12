#!/usr/bin/env python3
"""
Model dosyalarını AWS S3'ten indirme scripti
"""

import os
import boto3
from botocore.exceptions import NoCredentialsError

def download_from_s3(bucket_name, s3_key, local_path):
    """S3'ten dosya indir"""
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_key, local_path)
        return True
    except NoCredentialsError:
        print("AWS credentials bulunamadı")
        return False
    except Exception as e:
        print(f"S3 indirme hatası: {e}")
        return False

def download_models():
    """Model dosyalarını S3'ten indir"""
    
    # Dizinleri oluştur
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/deep_learning", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # S3 bucket ve dosya listesi
    bucket_name = "your-models-bucket"
    model_files = {
        "models/tfidf_vectorizer.pkl": "models/tfidf_vectorizer.pkl",
        "models/dolar_skor_rf_model.pkl": "models/dolar_skor_rf_model.pkl",
        "data/glove.6B.100d.txt": "data/glove.6B.100d.txt",
        # Diğer dosyalar...
    }
    
    print("📥 Model dosyaları S3'ten indiriliyor...")
    
    for local_path, s3_key in model_files.items():
        try:
            print(f"İndiriliyor: {local_path}")
            if download_from_s3(bucket_name, s3_key, local_path):
                print(f"✅ {local_path} indirildi")
            else:
                print(f"❌ {local_path} indirilemedi")
        except Exception as e:
            print(f"❌ {local_path} hatası: {e}")
    
    print("🎉 Model indirme tamamlandı!")

if __name__ == "__main__":
    download_models()
