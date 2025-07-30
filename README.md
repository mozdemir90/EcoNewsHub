# Haber Skor Tahmini Projesi

Bu proje, haber metinlerinin finansal varlıklar (Dolar, Altın, Borsa, Bitcoin) üzerindeki etkisini tahmin eden bir makine öğrenmesi sistemidir.

## 🚀 Özellikler

### Supervised Learning Modelleri
- **TF-IDF + Regresyon Modelleri**
  - Random Forest
  - Support Vector Machine (SVM)
  - Artificial Neural Network (ANN)
  - AdaBoost
  - Naive Bayes

- **Word2Vec + Regresyon Modelleri**
  - Random Forest
  - SVM
  - ANN
  - AdaBoost

- **GloVe + Regresyon Modelleri**
  - Random Forest
  - SVM
  - ANN
  - AdaBoost

### Deep Learning Modelleri
- **1D CNN** - Convolutional Neural Network
- **LSTM** - Long Short-Term Memory
- **CNN + LSTM** - Hibrit model

## 📁 Proje Yapısı

```
newsFetch/
├── app.py                          # Flask web uygulaması
├── trainTF-IDF.py                  # TF-IDF modelleri eğitimi
├── trainWord2Vec_GloVe.py          # Word2Vec/GloVe modelleri eğitimi
├── trainDeepLearning.py            # Deep Learning modelleri eğitimi
├── compare_models.py               # Model karşılaştırma scripti
├── labelNews.py                    # Haber etiketleme ve skorlama
├── news_fetcher.py                 # Haber çekme scripti
├── prep.py                         # Veri ön işleme
├── requirements.txt                # Gerekli kütüphaneler
├── data/                          # Veri dosyaları
│   ├── training_data.xlsx         # Eğitim verisi
│   ├── analiz_sonuclari2.xlsx    # Test verisi
│   └── ...
├── models/                        # Eğitilmiş modeller
│   ├── supervised/                # Supervised learning modelleri
│   └── deep_learning/            # Deep learning modelleri
└── templates/                     # Web arayüzü
    ├── index.html
    └── ekle.html
```

## 🛠️ Kurulum

1. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Veri hazırlığı:**
```bash
python labelNews.py
python trainTF-IDF.py
python trainWord2Vec_GloVe.py
python trainDeepLearning.py
```

3. **Web uygulamasını başlatın:**
```bash
python app.py
```

## 📊 Model Karşılaştırması

Tüm modelleri karşılaştırmak için:
```bash
python compare_models.py
```

Bu script şunları oluşturur:
- `data/model_comparison.xlsx` - Karşılaştırma tablosu
- `data/model_comparison_heatmap.png` - Performans heatmap'i
- `data/r2_comparison.png` - R² skorları grafiği
- `data/model_comparison_report.md` - Detaylı rapor

## 🌐 Web Arayüzü

- **Ana Sayfa:** Haber tahmini yapma
- **Eğitim Setine Ekle:** Yeni haber ve skor ekleme
- **Model Seçenekleri:**
  - TF-IDF, Word2Vec, GloVe (Supervised Learning)
  - Deep Learning (CNN, LSTM, CNN+LSTM)

## 📈 Performans Metrikleri

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared)
- **RMSE** (Root Mean Squared Error)

## 🔧 Kullanım

1. Web arayüzünde haber metnini girin
2. Yöntem seçin (TF-IDF, Word2Vec, GloVe, Deep Learning)
3. Model seçin
4. "Tahmin Et" butonuna tıklayın
5. Sonuçları görüntüleyin

## 📝 Skor Sistemi

- **0:** Güçlü düşüş etkisi
- **1:** Düşüş etkisi
- **2:** Hafif düşüş etkisi
- **3:** Nötr/etkisiz
- **4:** Yükseliş etkisi
- **5:** Güçlü yükseliş etkisi

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 