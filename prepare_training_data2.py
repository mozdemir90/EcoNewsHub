#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yeni oluşturulan dosyayı standart training_data formatına getirme scripti
"""

import pandas as pd
import logging
import re
import unicodedata
from datetime import datetime

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def normalize_text(text):
    """
    Metni normalize et
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Unicode normalize
    text = unicodedata.normalize('NFKC', str(text))
    
    # Gereksiz boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_training_data2():
    """
    Yeni oluşturulan dosyayı standart training_data formatına getir
    """
    try:
        # Dosya yolları
        input_file = "data/training_data_with_glove_rf_20250828_163835.xlsx"
        output_file = "data/training_data2.xlsx"
        
        logging.info(f"📖 {input_file} dosyası okunuyor...")
        df = pd.read_excel(input_file)
        
        logging.info(f"📊 Başlangıç satır sayısı: {len(df)}")
        
        # 1. Eksik değerleri doldur
        logging.info("🔧 Eksik değerler dolduruluyor...")
        
        # ozet sütununu doldur (content'in ilk 100 karakteri)
        df['ozet'] = df['ozet'].fillna('')
        for idx, row in df.iterrows():
            if row['ozet'] == '':
                content = str(row['content'])
                df.at[idx, 'ozet'] = content[:100] + '...' if len(content) > 100 else content
        
        # content_norm sütununu doldur
        df['content_norm'] = df['content_norm'].fillna('')
        for idx, row in df.iterrows():
            if row['content_norm'] == '':
                df.at[idx, 'content_norm'] = normalize_text(row['content'])
        
        # text sütununu doldur
        df['text'] = df['text'].fillna('')
        for idx, row in df.iterrows():
            if row['text'] == '':
                df.at[idx, 'text'] = normalize_text(row['content'])
        
        # added_by sütununu doldur
        df['added_by'] = df['added_by'].fillna('glove_rf_merge')
        
        # added_date sütununu doldur
        df['added_date'] = df['added_date'].fillna(pd.Timestamp.now())
        
        # 2. Veri tiplerini düzenle
        logging.info("🔧 Veri tipleri düzenleniyor...")
        
        # Skor sütunlarını int'e çevir
        score_columns = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        for col in score_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3).astype(int)
        
        # language sütununu düzenle
        df['language'] = df['language'].fillna('tr')
        
        # 3. Veri kalitesi kontrolü
        logging.info("🔍 Veri kalitesi kontrol ediliyor...")
        
        # Content uzunluğu kontrolü
        short_content = df[df['content'].str.len() < 10]
        if len(short_content) > 0:
            logging.warning(f"⚠️ {len(short_content)} kısa content bulundu")
        
        # Skor aralığı kontrolü (1-5)
        for col in score_columns:
            invalid_scores = df[(df[col] < 1) | (df[col] > 5)]
            if len(invalid_scores) > 0:
                logging.warning(f"⚠️ {len(invalid_scores)} geçersiz {col} bulundu")
                # Geçersiz skorları 3'e çevir
                df.loc[(df[col] < 1) | (df[col] > 5), col] = 3
        
        # 4. Duplicate content'leri kontrol et
        logging.info("🔍 Duplicate content'ler kontrol ediliyor...")
        duplicates = df[df.duplicated(subset=['content'], keep=False)]
        if len(duplicates) > 0:
            logging.warning(f"⚠️ {len(duplicates)} duplicate content bulundu")
            # Duplicate'leri kaldır (ilkini tut)
            df = df.drop_duplicates(subset=['content'], keep='first')
            logging.info(f"📊 Duplicate'ler kaldırıldı, yeni satır sayısı: {len(df)}")
        
        # 5. Son kontrol ve kaydetme
        logging.info("💾 Dosya kaydediliyor...")
        
        # Sütun sırasını düzenle (orijinal training_data ile aynı)
        column_order = ['content', 'ozet', 'language', 'dolar_skor', 'altin_skor', 
                       'borsa_skor', 'bitcoin_skor', 'content_norm', 'text', 
                       'added_by', 'added_date']
        df = df[column_order]
        
        # Dosyayı kaydet
        df.to_excel(output_file, index=False)
        
        # Özet bilgiler
        print("\n" + "="*60)
        print("📈 TRAINING_DATA2 HAZIRLAMA ÖZETİ")
        print("="*60)
        print(f"📊 Toplam Satır Sayısı: {len(df)}")
        print(f"📋 Sütun Sayısı: {len(df.columns)}")
        print(f"🔍 Skor Dağılımı:")
        for col in score_columns:
            print(f"   {col}: {df[col].value_counts().sort_index().to_dict()}")
        print(f"💾 Çıktı Dosyası: {output_file}")
        print("="*60)
        
        logging.info(f"✅ Training_data2 hazırlandı!")
        logging.info(f"📁 Çıktı dosyası: {output_file}")
        
        return output_file
        
    except FileNotFoundError as e:
        logging.error(f"❌ Dosya bulunamadı: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    prepare_training_data2()



