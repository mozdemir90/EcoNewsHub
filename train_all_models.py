#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tüm modelleri training_data4.xlsx ile eğiten otomatik script
"""

import subprocess
import sys
import time
import logging
from datetime import datetime

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_training_script(script_name, description):
    """Eğitim scriptini çalıştır"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, timeout=1800)  # 30 dakika timeout
        
        if result.returncode == 0:
            print(f"✅ {description} başarıyla tamamlandı!")
            print(f"⏱️  Süre: {time.time() - start_time:.2f} saniye")
            return True
        else:
            print(f"❌ {description} başarısız!")
            print(f"Hata: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} zaman aşımına uğradı (30 dakika)")
        return False
    except Exception as e:
        print(f"❌ {description} çalıştırılırken hata: {e}")
        return False

def main():
    """Ana eğitim süreci"""
    print("🎯 TÜM MODELLERİ EĞİTME SÜRECİ BAŞLIYOR")
    print("📊 Kullanılan veri seti: training_data4.xlsx")
    print(f"🕐 Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Eğitim scriptleri ve açıklamaları
    training_scripts = [
        ("trainTF-IDF.py", "TF-IDF Modelleri Eğitimi (RF, SVM, NB, AdaBoost, ANN)"),
        ("trainWord2Vec_GloVe.py", "Word2Vec & GloVe Modelleri Eğitimi"),
        ("trainDeepLearning.py", "Deep Learning Modelleri Eğitimi (CNN, LSTM, CNN+LSTM)")
    ]
    
    success_count = 0
    total_scripts = len(training_scripts)
    
    for script, description in training_scripts:
        if run_training_script(script, description):
            success_count += 1
        else:
            print(f"⚠️ {description} atlandı, diğer eğitimler devam ediyor...")
    
    # Sonuç raporu
    print(f"\n{'='*60}")
    print("📊 EĞİTİM SÜRECİ TAMAMLANDI")
    print(f"{'='*60}")
    print(f"✅ Başarılı: {success_count}/{total_scripts}")
    print(f"❌ Başarısız: {total_scripts - success_count}/{total_scripts}")
    print(f"🕐 Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_scripts:
        print("\n🎉 Tüm modeller başarıyla eğitildi!")
        print("📁 Eğitilen modeller:")
        print("   - models/tf-idf/")
        print("   - models/word2vec/")
        print("   - models/glove/")
        print("   - models/deeplearning/")
    else:
        print(f"\n⚠️ {total_scripts - success_count} eğitim başarısız oldu.")
        print("Lütfen hata mesajlarını kontrol edin.")
    
    return success_count == total_scripts

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


