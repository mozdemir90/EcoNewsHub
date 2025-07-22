# Ekonomik Haber Analiz ve Tahmin Sistemi

Bu proje, haberlerin finansal varlıklar (Dolar, Altın, Borsa, Bitcoin) üzerindeki etkisini otomatik olarak analiz eden, skorlayan ve makine öğrenmesiyle tahmin eden bir sistemdir. Hem anahtar kelime tabanlı hem de makine öğrenmesi tabanlı (TF-IDF, Word2Vec, GloVe) modellemeler içerir. Web arayüzü ile kullanıcılar haber girişi yapabilir, model ve yöntem seçip tahmin alabilir, yeni haberleri eğitim setine ekleyebilir.

## Genel Akış
- **Haber Toplama & Temizleme:** Farklı kaynaklardan haberler toplanır, temizlenir ve özetlenir.
- **Haber Analizi & Skorlama:** Varlık bazlı anahtar kelime ve cümle analizi ile skorlar atanır.
- **Vektörizasyon:** TF-IDF, Word2Vec veya GloVe ile metinler vektörleştirilir.
- **Model Eğitimi:** Her yöntem için farklı regresyon modelleri (Random Forest, SVM, AdaBoost, ANN, Naive Bayes) eğitilir.
- **Tahmin & Sonuç:** Test seti için tahminler yapılır, sonuçlar kaydedilir.
- **Web Arayüzü:** Kullanıcılar haber girip model/yöntem seçerek tahmin alabilir, yeni haberleri eğitim setine ekleyebilir.

Daha detaylı akış için: [PROJECT_FLOW.md](PROJECT_FLOW.md)

## Temel Dosyalar
- `app.py` : Flask tabanlı web arayüzü, model yükleme ve tahmin.
- `trainTF-IDF.py` : TF-IDF tabanlı model eğitimi ve tahmin.
- `trainWord2Vec_GloVe.py` : Word2Vec ve GloVe tabanlı model eğitimi ve tahmin.
- `labelNews.py` : Anahtar kelime ve cümle bazlı varlık skorlama, özetleme, veri temizliği.
- `sentiment2LAbel.py` : Alternatif/ayrıntılı duygu ve varlık analizi.
- `prep.py` : Ham haber verisi temizleme, stopword çıkarma, etiketleme.
- `news_fetcher.py` : Farklı kaynaklardan haber çekme (scraper).
- `requirements.txt` : Gerekli Python paketleri.

## Kurulum
1. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
   pip install gensim langdetect flask joblib
   ```
2. (GloVe için) `data/glove.6B.100d.txt` dosyasını indirip ilgili klasöre koyun.

## Kullanım
- **Model Eğitimi:**
  - TF-IDF: `python trainTF-IDF.py`
  - Word2Vec/GloVe: `python trainWord2Vec_GloVe.py`
- **Web Arayüzü:**
  ```bash
  python app.py
  # Tarayıcıda: http://localhost:5050
  ```
- **Haber Temizleme/Özetleme:**
  - `python labelNews.py` veya `python prep.py`

## Notlar
- Eğitim seti: `data/training_data.xlsx` (duplicate özetler otomatik engellenir)
- Test seti: `data/analiz_sonuclari2.xlsx`
- Sonuçlar: `data/analiz_sonuclari2_tahminli_TF-IDF.xlsx`, `data/analiz_sonuclari2_tahminli_w2v.xlsx`, `data/analiz_sonuclari2_tahminli_glove.xlsx`
- Modeller: `models/` klasöründe saklanır.
- Loglar: `logs/` klasöründe tutulur (isteğe bağlı, Github'a atılmaz).

## Gereksiz Dosyalar (Github'a eklenmemeli)
- `logs/`, `output/`, `models/`, `data/*.xlsx`, `data/*.csv`, `.DS_Store`

## Lisans
MIT 