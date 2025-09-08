#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GloVe + Random Forest sonuçlarını training_data ile birleştirme scripti
"""

import pandas as pd
import logging
from datetime import datetime

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def merge_glove_rf_results():
    """
    haber_degerlendirme_sonuclari.xlsx'den GloVe + RF sonuçlarını alıp
    training_data2.xlsx ile birleştirir
    """
    try:
        # Dosya yolları
        results_file = "haber_degerlendirme_sonuclari.xlsx"
        training_file = "data/training_data2.xlsx"
        output_file = f"data/training_data_with_glove_rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        logging.info(f"📖 {results_file} dosyası okunuyor...")
        results_df = pd.read_excel(results_file)
        
        logging.info(f"📖 {training_file} dosyası okunuyor...")
        training_df = pd.read_excel(training_file)
        
        # GloVe + RF sütunlarını seç
        glove_rf_columns = ['content'] + [col for col in results_df.columns if col.startswith('GloVe + RF_')]
        logging.info(f"🔍 GloVe + RF sütunları seçiliyor: {len(glove_rf_columns)} sütun")
        
        # GloVe + RF sonuçlarını al
        glove_rf_results = results_df[glove_rf_columns].copy()
        
        # Sütun isimlerini düzenle (GloVe + RF_ prefix'ini kaldır ve yeni isimler ver)
        new_columns = ['content']
        for col in glove_rf_columns[1:]:  # content hariç
            new_name = col.replace('GloVe + RF_', 'glove_rf_')
            new_columns.append(new_name)
        
        glove_rf_results.columns = new_columns
        logging.info(f"📝 Sütun isimleri düzenlendi: {new_columns}")
        
        # Training data'yı kopyala
        merged_df = training_df.copy()
        
        # GloVe + RF sonuçlarını training data'ya ekle
        logging.info("🔗 GloVe + RF sonuçları training data'ya ekleniyor...")
        
        # Content'e göre merge yap
        temp_merge = pd.merge(
            merged_df[['content']], 
            glove_rf_results, 
            on='content', 
            how='left'
        )
        
        # Eşleşen satırları güncelle
        for idx, row in temp_merge.iterrows():
            if pd.notna(row['glove_rf_Dolar']):
                # Training data'da bu content'in index'ini bul
                training_idx = merged_df[merged_df['content'] == row['content']].index[0]
                # Skorları güncelle
                merged_df.at[training_idx, 'dolar_skor'] = row['glove_rf_Dolar']
                merged_df.at[training_idx, 'altin_skor'] = row['glove_rf_Altın']
                merged_df.at[training_idx, 'borsa_skor'] = row['glove_rf_Borsa']
                merged_df.at[training_idx, 'bitcoin_skor'] = row['glove_rf_Bitcoin']
        
        # Şimdi tüm GloVe + RF sonuçlarını ekle
        logging.info("🔗 Tüm GloVe + RF sonuçları ekleniyor...")
        
        # Training data'da olmayan content'leri bul
        training_contents = set(training_df['content'])
        all_results_contents = set(glove_rf_results['content'])
        new_contents = all_results_contents - training_contents
        
        logging.info(f"📊 Training data'da olmayan yeni content sayısı: {len(new_contents)}")
        
        # Yeni content'ler için yeni satırlar oluştur
        new_rows = []
        for content in new_contents:
            result_row = glove_rf_results[glove_rf_results['content'] == content].iloc[0]
            
            # Yeni satır oluştur (training data formatında)
            new_row = {
                'content': content,
                'ozet': '',  # Boş bırak
                'language': 'tr',  # Varsayılan
                'dolar_skor': result_row['glove_rf_Dolar'],
                'altin_skor': result_row['glove_rf_Altın'],
                'borsa_skor': result_row['glove_rf_Borsa'],
                'bitcoin_skor': result_row['glove_rf_Bitcoin'],
                'content_norm': content,  # Aynı content
                'text': content,  # Aynı content
                'added_by': 'glove_rf_merge',  # Kaynak belirt
                'added_date': pd.Timestamp.now()  # Şu anki tarih
            }
            new_rows.append(new_row)
        
        # Yeni satırları DataFrame'e ekle
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            merged_df = pd.concat([merged_df, new_df], ignore_index=True)
            logging.info(f"📊 {len(new_rows)} yeni satır eklendi")
        
        # Eşleşen ve eşleşmeyen satırları say
        matched_rows = merged_df[merged_df['dolar_skor'].notna()].shape[0]
        unmatched_rows = merged_df[merged_df['dolar_skor'].isna()].shape[0]
        
        # Birleştirme sonuçlarını kontrol et
        total_rows = len(merged_df)
        logging.info(f"📊 Toplam satır: {total_rows}")
        logging.info(f"📊 Eşleşen satır: {matched_rows}")
        logging.info(f"📊 Eşleşmeyen satır: {unmatched_rows}")
        logging.info(f"📊 Eşleşme oranı: {matched_rows/total_rows*100:.2f}%")
        
        # Sonuçları kaydet
        logging.info(f"💾 Sonuçlar {output_file} dosyasına kaydediliyor...")
        merged_df.to_excel(output_file, index=False)
        
        logging.info(f"✅ Birleştirme tamamlandı!")
        logging.info(f"📁 Çıktı dosyası: {output_file}")
        
        # Özet bilgiler
        print("\n" + "="*60)
        print("📈 BİRLEŞTİRME ÖZETİ")
        print("="*60)
        print(f"📰 Training Data Satır Sayısı: {len(training_df)}")
        print(f"📊 GloVe + RF Sonuç Satır Sayısı: {len(glove_rf_results)}")
        print(f"🔗 Birleştirilmiş Satır Sayısı: {len(merged_df)}")
        print(f"✅ Eşleşen Satır Sayısı: {matched_rows}")
        print(f"❌ Eşleşmeyen Satır Sayısı: {unmatched_rows}")
        print(f"📊 Eşleşme Oranı: {matched_rows/total_rows*100:.2f}%")
        print(f"💾 Çıktı Dosyası: {output_file}")
        print("="*60)
        
        return output_file
        
    except FileNotFoundError as e:
        logging.error(f"❌ Dosya bulunamadı: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    merge_glove_rf_results()
