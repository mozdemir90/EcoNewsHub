#!/usr/bin/env python3
"""
Toplu Haber Değerlendirme Scripti
haberler_detayli_lang_tarih1.xlsx dosyasındaki haberleri tüm modellerle değerlendirir
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# app.py'den gerekli fonksiyonları import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import (
        predict_deep_learning, 
        predict_hybrid, 
        get_vectorizer_tfidf, 
        model_files,
        tokenize,
        get_sentence_vector,
        get_sentence_vector_glove,
        w2v_model,
        glove_vectors,
        TENSORFLOW_AVAILABLE
    )
    logger.info("✅ app.py fonksiyonları başarıyla import edildi")
except ImportError as e:
    logger.error(f"❌ app.py import hatası: {e}")
    sys.exit(1)

def evaluate_single_news(text, method="tfidf", model_name="rf", dl_model="cnn", hybrid_weights=(0.4, 0.6)):
    """
    Tek bir haberi belirtilen yöntemle değerlendir
    
    Args:
        text: Haber metni
        method: Değerlendirme yöntemi (tfidf, w2v, glove, deep_learning, hybrid)
        model_name: Model adı
        dl_model: Deep Learning model adı
        hybrid_weights: Hibrit ağırlıkları
    
    Returns:
        Dict: Varlık skorları
    """
    try:
        if not text or pd.isna(text) or len(str(text).strip()) < 4:
            return {"Dolar": 3, "Altın": 3, "Borsa": 3, "Bitcoin": 3}
        
        text = str(text).strip()
        
        if method == "tfidf":
            vectorizer = get_vectorizer_tfidf()
            if vectorizer is None:
                return {"Dolar": "TF-IDF Yok", "Altın": "TF-IDF Yok", "Borsa": "TF-IDF Yok", "Bitcoin": "TF-IDF Yok"}
            
            X = vectorizer.transform([text])
            if model_name not in model_files.get("tfidf", {}):
                return {"Dolar": "Model Yok", "Altın": "Model Yok", "Borsa": "Model Yok", "Bitcoin": "Model Yok"}
            
            modeller = model_files["tfidf"][model_name]
            X_input = X.toarray() if model_name == "nb" else X
            
            return {
                "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X_input)[0]))),
                "Altın": min(5, max(1, round(modeller["Altın"].predict(X_input)[0]))),
                "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X_input)[0]))),
                "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X_input)[0])))
            }
        
        elif method == "w2v":
            if w2v_model is None:
                return {"Dolar": "Word2Vec Yok", "Altın": "Word2Vec Yok", "Borsa": "Word2Vec Yok", "Bitcoin": "Word2Vec Yok"}
            
            tokens = tokenize(text)
            X = np.array([get_sentence_vector(tokens, w2v_model)])
            
            if model_name not in model_files.get("w2v", {}):
                return {"Dolar": "Model Yok", "Altın": "Model Yok", "Borsa": "Model Yok", "Bitcoin": "Model Yok"}
            
            modeller = model_files["w2v"][model_name]
            
            return {
                "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X)[0]))),
                "Altın": min(5, max(1, round(modeller["Altın"].predict(X)[0]))),
                "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X)[0]))),
                "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X)[0])))
            }
        
        elif method == "glove":
            if glove_vectors is None:
                return {"Dolar": "GloVe Yok", "Altın": "GloVe Yok", "Borsa": "GloVe Yok", "Bitcoin": "GloVe Yok"}
            
            tokens = tokenize(text)
            X = np.array([get_sentence_vector_glove(tokens, glove_vectors)])
            
            if model_name not in model_files.get("glove", {}):
                return {"Dolar": "Model Yok", "Altın": "Model Yok", "Borsa": "Model Yok", "Bitcoin": "Model Yok"}
            
            modeller = model_files["glove"][model_name]
            
            return {
                "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X)[0]))),
                "Altın": min(5, max(1, round(modeller["Altın"].predict(X)[0]))),
                "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X)[0]))),
                "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X)[0])))
            }
        
        elif method == "deep_learning":
            if not TENSORFLOW_AVAILABLE:
                return {"Dolar": "TensorFlow Yok", "Altın": "TensorFlow Yok", "Borsa": "TensorFlow Yok", "Bitcoin": "TensorFlow Yok"}
            
            result = predict_deep_learning(text, dl_model)
            if result is not None:
                return result
            else:
                return {"Dolar": "DL Hatası", "Altın": "DL Hatası", "Borsa": "DL Hatası", "Bitcoin": "DL Hatası"}
        
        elif method == "hybrid":
            result = predict_hybrid(text, dl_model, model_name, "tfidf", hybrid_weights)
            if result is not None:
                return result["prediction"]
            else:
                return {"Dolar": "Hibrit Hatası", "Altın": "Hibrit Hatası", "Borsa": "Hibrit Hatası", "Bitcoin": "Hibrit Hatası"}
        
        else:
            return {"Dolar": "Geçersiz Yöntem", "Altın": "Geçersiz Yöntem", "Borsa": "Geçersiz Yöntem", "Bitcoin": "Geçersiz Yöntem"}
    
    except Exception as e:
        logger.error(f"Haber değerlendirme hatası: {e}")
        return {"Dolar": "Hata", "Altın": "Hata", "Borsa": "Hata", "Bitcoin": "Hata"}

def batch_evaluate_news():
    """
    Tüm haberleri farklı modellerle değerlendir
    """
    input_file = "data/haberler_detayli_lang_tarih1.xlsx"
    
    # Dosya kontrolü
    if not os.path.exists(input_file):
        logger.error(f"❌ {input_file} dosyası bulunamadı!")
        return
    
    try:
        # Veri setini yükle
        logger.info(f"📖 {input_file} dosyası yükleniyor...")
        df = pd.read_excel(input_file)
        
        if 'content' not in df.columns:
            logger.error("❌ 'content' sütunu bulunamadı!")
            return
        
        logger.info(f"✅ {len(df)} haber yüklendi")
        
        # Değerlendirme yöntemleri
        methods = [
            ("tfidf", "rf", "Random Forest"),
            ("tfidf", "svm", "SVM"),
            ("tfidf", "nb", "Naive Bayes"),
            ("tfidf", "ann", "ANN"),
            ("tfidf", "ada", "AdaBoost"),
            ("w2v", "rf", "Word2Vec + RF"),
            ("w2v", "svm", "Word2Vec + SVM"),
            ("w2v", "ann", "Word2Vec + ANN"),
            ("w2v", "ada", "Word2Vec + AdaBoost"),
            ("glove", "rf", "GloVe + RF"),
            ("glove", "svm", "GloVe + SVM"),
            ("glove", "ann", "GloVe + ANN"),
            ("glove", "ada", "GloVe + AdaBoost"),
            ("deep_learning", "cnn", "CNN"),
            ("deep_learning", "lstm", "LSTM"),
            ("deep_learning", "cnn_lstm", "CNN+LSTM"),
            ("hybrid", "rf", "Hibrit (40-60)"),
        ]
        
        # Sonuç DataFrame'i oluştur
        results_df = df[['content']].copy()
        
        # Her yöntem için değerlendirme yap
        for method, model_name, method_name in methods:
            logger.info(f"🔍 {method_name} ile değerlendiriliyor...")
            
            # Deep Learning için özel parametreler
            dl_model = model_name if method == "deep_learning" else "cnn"
            hybrid_weights = (0.4, 0.6) if method == "hybrid" else None
            
            # Her haber için değerlendirme
            scores = []
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"   {idx}/{len(df)} haber işlendi...")
                
                content = row['content']
                score = evaluate_single_news(content, method, model_name, dl_model, hybrid_weights)
                scores.append(score)
            
            # Sonuçları DataFrame'e ekle
            for asset in ["Dolar", "Altın", "Borsa", "Bitcoin"]:
                col_name = f"{method_name}_{asset}"
                results_df[col_name] = [score[asset] for score in scores]
        
        # Sonuç dosyasını kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"haber_degerlendirme_sonuclari_{timestamp}.xlsx"
        
        logger.info(f"💾 Sonuçlar {output_file} dosyasına kaydediliyor...")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='Tüm Sonuçlar', index=False)
            
            # Özet sayfası oluştur
            summary_data = []
            for method, model_name, method_name in methods:
                for asset in ["Dolar", "Altın", "Borsa", "Bitcoin"]:
                    col_name = f"{method_name}_{asset}"
                    if col_name in results_df.columns:
                        values = pd.to_numeric(results_df[col_name], errors='coerce')
                        valid_values = values.dropna()
                        if len(valid_values) > 0:
                            summary_data.append({
                                'Yöntem': method_name,
                                'Varlık': asset,
                                'Ortalama': valid_values.mean(),
                                'Medyan': valid_values.median(),
                                'Std': valid_values.std(),
                                'Min': valid_values.min(),
                                'Max': valid_values.max(),
                                'Geçerli Veri': len(valid_values),
                                'Toplam Veri': len(values)
                            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Özet İstatistikler', index=False)
        
        logger.info(f"✅ Değerlendirme tamamlandı! Sonuçlar: {output_file}")
        logger.info(f"📊 Toplam {len(df)} haber, {len(methods)} yöntemle değerlendirildi")
        
        # Özet bilgileri yazdır
        print("\n" + "="*60)
        print("📈 DEĞERLENDİRME ÖZETİ")
        print("="*60)
        print(f"📰 Toplam Haber: {len(df)}")
        print(f"🔧 Kullanılan Yöntem: {len(methods)}")
        print(f"💾 Çıktı Dosyası: {output_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Toplu değerlendirme hatası: {e}")
        raise

if __name__ == "__main__":
    print("🚀 Toplu Haber Değerlendirme Başlıyor...")
    print("="*60)
    batch_evaluate_news()
