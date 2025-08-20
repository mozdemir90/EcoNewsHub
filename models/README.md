# Models Directory Organization

Bu klasör, farklı makine öğrenmesi modellerini kategorilere göre organize eder.

## Klasör Yapısı

### 📁 tf-idf/
TF-IDF tabanlı tüm modeller:
- **tfidf_vectorizer.pkl**: TF-IDF vektörizer modeli
- **{varlik}_rf_model.pkl**: Random Forest modelleri (dollar, gold, bist100, bitcoin)
- **{varlik}_svm_model.pkl**: Support Vector Machine modelleri
- **{varlik}_nb_model.pkl**: Naive Bayes modelleri
- **{varlik}_ada_model.pkl**: AdaBoost modelleri
- **{varlik}_ann_model.pkl**: Artificial Neural Network modelleri

### 📁 word2vec/
Word2Vec tabanlı modeller:
- **word2vec_model.model**: Word2Vec kelime gömme modeli
- **{varlik}_rf_w2v_model.pkl**: Random Forest + Word2Vec modelleri
- **{varlik}_svm_w2v_model.pkl**: SVM + Word2Vec modelleri
- **{varlik}_ann_w2v_model.pkl**: ANN + Word2Vec modelleri
- **{varlik}_ada_w2v_model.pkl**: AdaBoost + Word2Vec modelleri

### 📁 glove/
GloVe tabanlı modeller:
- **{varlik}_rf_glove_model.pkl**: Random Forest + GloVe modelleri
- **{varlik}_svm_glove_model.pkl**: SVM + GloVe modelleri
- **{varlik}_ann_glove_model.pkl**: ANN + GloVe modelleri
- **{varlik}_ada_glove_model.pkl**: AdaBoost + GloVe modelleri

### 📁 traditional_ml/
⚠️ **BOŞ KLASÖR** - Tüm modeller ilgili kategorilere taşındı

### 📁 deeplearning/
Derin öğrenme modelleri:
- **cnn_model.h5**: Convolutional Neural Network modeli
- **cnn_best.h5**: CNN en iyi modeli
- **lstm_model.h5**: Long Short-Term Memory modeli
- **lstm_best.h5**: LSTM en iyi modeli
- **cnn_lstm_model.h5**: CNN-LSTM hibrit modeli
- **cnn_lstm_best.h5**: CNN-LSTM en iyi modeli
- **tokenizer.pkl**: Metin tokenizer'ı

### 📁 backup_organized/
Eski model yedekleri:
- **backup_current/**: Mevcut yedekler
- **backup_old_models/**: Eski model yedekleri
- **backup_other_models/**: Diğer model yedekleri

## Model Kategorileri

1. **TF-IDF Modelleri**: Metin verilerini TF-IDF vektörlerine dönüştürerek geleneksel ML algoritmaları kullanır
2. **Word2Vec Modelleri**: Kelime gömme teknikleri ile geleneksel ML algoritmaları kullanır
3. **GloVe Modelleri**: Global Vectors for Word Representation ile geleneksel ML algoritmaları kullanır
4. **Deep Learning Modelleri**: Derin öğrenme mimarileri (CNN, LSTM, CNN-LSTM)

## Kullanım

Modeller şu dosyalarda yüklenir:
- `app.py`: Web uygulaması için model yükleme
- `telegram_bot.py`: Telegram bot için model yükleme
- `telegram_interactive_bot.py`: İnteraktif Telegram bot için model yükleme

## Eğitim

Modeller şu dosyalarda eğitilir:
- `trainTF-IDF.py`: TF-IDF tabanlı modeller
- `trainWord2Vec_GloVe.py`: Word2Vec ve GloVe tabanlı modeller
- `trainDeepLearning.py`: Derin öğrenme modelleri

## Model Performansı

Son karşılaştırma sonuçlarına göre:
- **Bitcoin**: TF-IDF en iyi performans (R² = 0.1745)
- **Dollar**: Word2Vec en iyi performans (R² = -0.3599)
- **Gold**: GloVe en iyi performans (R² = -0.0304)
- **BIST100**: GloVe en iyi performans (R² = -0.0650)

## Güncelleme Tarihi

Son güncelleme: 20 Ağustos 2025
- TF-IDF modelleri `traditional_ml/` klasöründen `tf-idf/` klasörüne taşındı
- Tüm model yolları güncellendi
- `app.py` ve diğer dosyalar yeni yapıya uyarlandı
