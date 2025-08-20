from flask import Flask, render_template, request
import joblib
import os
import pandas as pd
import logging
from langdetect import detect, LangDetectException
import unicodedata
import re
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from functools import lru_cache
import nltk
from nltk.tokenize import sent_tokenize

# NLTK verilerini indir (eğer yoksa)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Model dosya yolları
GLOVE_PATH = "data/glove.6B.100d.txt"

# Deep Learning modelleri
DL_MODELS = {}
DL_TOKENIZER = None



# Deep Learning modellerini yükle (varsa)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow bulunamadı, Deep Learning modelleri kullanılamayacak")
    TENSORFLOW_AVAILABLE = False

# Tokenizer yükle (TensorFlow olsun ya da olmasın)
try:
    if os.path.exists("models/deeplearning/tokenizer.pkl"):
        DL_TOKENIZER = joblib.load("models/deeplearning/tokenizer.pkl")
        print("✅ Deep Learning tokenizer yüklendi")
    else:
        DL_TOKENIZER = None
        print("⚠️ Deep Learning tokenizer dosyası bulunamadı")
except Exception as e:
    DL_TOKENIZER = None
    print(f"❌ Deep Learning tokenizer yüklenemedi: {e}")

# Modelleri yükle (sadece TensorFlow varsa)
if TENSORFLOW_AVAILABLE:
    dl_model_files = {
        'cnn': 'models/deeplearning/cnn_model.h5',
        'lstm': 'models/deeplearning/lstm_model.h5',
        'cnn_lstm': 'models/deeplearning/cnn_lstm_model.h5'
    }
    
    for model_name, model_path in dl_model_files.items():
        if os.path.exists(model_path):
            try:
                DL_MODELS[model_name] = load_model(model_path, compile=False)
                print(f"✅ {model_name} modeli yüklendi")
            except Exception as e:
                print(f"❌ {model_name} modeli yüklenemedi: {e}")
                # Model yüklenemezse, alternatif olarak best model'i dene
                best_model_path = model_path.replace('_model.h5', '_best.h5')
                if os.path.exists(best_model_path):
                    try:
                        DL_MODELS[model_name] = load_model(best_model_path, compile=False)
                        print(f"✅ {model_name} best modeli yüklendi")
                    except Exception as e2:
                        print(f"❌ {model_name} best modeli de yüklenemedi: {e2}")
        else:
            print(f"⚠️ {model_name} model dosyası bulunamadı: {model_path}")
    
    print(f"Deep Learning modelleri: {list(DL_MODELS.keys())}")
else:
    print("TensorFlow olmadığı için Deep Learning modelleri yüklenmedi")



glove_vectors = None
if os.path.exists(GLOVE_PATH):
    glove_vectors = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False, no_header=True)
else:
    print(f"UYARI: {GLOVE_PATH} bulunamadı, GloVe ile tahmin yapılamaz.")

# Model dosyalarını güvenli şekilde yükle
def load_model_safe(model_path):
    """Model dosyasını güvenli şekilde yükle"""
    try:
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            print(f"⚠️ Model dosyası bulunamadı: {model_path}")
            return None
    except Exception as e:
        print(f"❌ Model yükleme hatası {model_path}: {e}")
        return None

model_files = {
    "tfidf": {
        "rf": {
            "Dolar": load_model_safe("models/tf-idf/dolar_skor_rf_model.pkl"),
            "Altın": load_model_safe("models/tf-idf/altin_skor_rf_model.pkl"),
            "Borsa": load_model_safe("models/tf-idf/borsa_skor_rf_model.pkl"),
            "Bitcoin": load_model_safe("models/tf-idf/bitcoin_skor_rf_model.pkl"),
        },
        "ann": {
            "Dolar": load_model_safe("models/tf-idf/dolar_skor_ann_model.pkl"),
            "Altın": load_model_safe("models/tf-idf/altin_skor_ann_model.pkl"),
            "Borsa": load_model_safe("models/tf-idf/borsa_skor_ann_model.pkl"),
            "Bitcoin": load_model_safe("models/tf-idf/bitcoin_skor_ann_model.pkl"),
        },
        "svm": {
            "Dolar": load_model_safe("models/tf-idf/dolar_skor_svm_model.pkl"),
            "Altın": load_model_safe("models/tf-idf/altin_skor_svm_model.pkl"),
            "Borsa": load_model_safe("models/tf-idf/borsa_skor_svm_model.pkl"),
            "Bitcoin": load_model_safe("models/tf-idf/bitcoin_skor_svm_model.pkl"),
        },
        "nb": {
            "Dolar": load_model_safe("models/tf-idf/dolar_skor_nb_model.pkl"),
            "Altın": load_model_safe("models/tf-idf/altin_skor_nb_model.pkl"),
            "Borsa": load_model_safe("models/tf-idf/borsa_skor_nb_model.pkl"),
            "Bitcoin": load_model_safe("models/tf-idf/bitcoin_skor_nb_model.pkl"),
        },
        "ada": {
            "Dolar": load_model_safe("models/tf-idf/dolar_skor_ada_model.pkl"),
            "Altın": load_model_safe("models/tf-idf/altin_skor_ada_model.pkl"),
            "Borsa": load_model_safe("models/tf-idf/borsa_skor_ada_model.pkl"),
            "Bitcoin": load_model_safe("models/tf-idf/bitcoin_skor_ada_model.pkl"),
        }
    },
    "w2v": {
        "rf": {
            "Dolar": load_model_safe("models/word2vec/dolar_skor_rf_w2v_model.pkl"),
            "Altın": load_model_safe("models/word2vec/altin_skor_rf_w2v_model.pkl"),
            "Borsa": load_model_safe("models/word2vec/borsa_skor_rf_w2v_model.pkl"),
            "Bitcoin": load_model_safe("models/word2vec/bitcoin_skor_rf_w2v_model.pkl"),
        },
        "ann": {
            "Dolar": load_model_safe("models/word2vec/dolar_skor_ann_w2v_model.pkl"),
            "Altın": load_model_safe("models/word2vec/altin_skor_ann_w2v_model.pkl"),
            "Borsa": load_model_safe("models/word2vec/borsa_skor_ann_w2v_model.pkl"),
            "Bitcoin": load_model_safe("models/word2vec/bitcoin_skor_ann_w2v_model.pkl"),
        },
        "svm": {
            "Dolar": load_model_safe("models/word2vec/dolar_skor_svm_w2v_model.pkl"),
            "Altın": load_model_safe("models/word2vec/altin_skor_svm_w2v_model.pkl"),
            "Borsa": load_model_safe("models/word2vec/borsa_skor_svm_w2v_model.pkl"),
            "Bitcoin": load_model_safe("models/word2vec/bitcoin_skor_svm_w2v_model.pkl"),
        },
        "ada": {
            "Dolar": load_model_safe("models/word2vec/dolar_skor_ada_w2v_model.pkl"),
            "Altın": load_model_safe("models/word2vec/altin_skor_ada_w2v_model.pkl"),
            "Borsa": load_model_safe("models/word2vec/borsa_skor_ada_w2v_model.pkl"),
            "Bitcoin": load_model_safe("models/word2vec/bitcoin_skor_ada_w2v_model.pkl"),
        }
    },
    "glove": {
        "rf": {
            "Dolar": load_model_safe("models/glove/dolar_skor_rf_glove_model.pkl"),
            "Altın": load_model_safe("models/glove/altin_skor_rf_glove_model.pkl"),
            "Borsa": load_model_safe("models/glove/borsa_skor_rf_glove_model.pkl"),
            "Bitcoin": load_model_safe("models/glove/bitcoin_skor_rf_glove_model.pkl"),
        },
        "ann": {
            "Dolar": load_model_safe("models/glove/dolar_skor_ann_glove_model.pkl"),
            "Altın": load_model_safe("models/glove/altin_skor_ann_glove_model.pkl"),
            "Borsa": load_model_safe("models/glove/borsa_skor_ann_glove_model.pkl"),
            "Bitcoin": load_model_safe("models/glove/bitcoin_skor_ann_glove_model.pkl"),
        },
        "svm": {
            "Dolar": load_model_safe("models/glove/dolar_skor_svm_glove_model.pkl"),
            "Altın": load_model_safe("models/glove/altin_skor_svm_glove_model.pkl"),
            "Borsa": load_model_safe("models/glove/borsa_skor_svm_glove_model.pkl"),
            "Bitcoin": load_model_safe("models/glove/bitcoin_skor_svm_glove_model.pkl"),
        },
        "ada": {
            "Dolar": load_model_safe("models/glove/dolar_skor_ada_glove_model.pkl"),
            "Altın": load_model_safe("models/glove/altin_skor_ada_glove_model.pkl"),
            "Borsa": load_model_safe("models/glove/borsa_skor_ada_glove_model.pkl"),
            "Bitcoin": load_model_safe("models/glove/bitcoin_skor_ada_glove_model.pkl"),
        }
    }
}

# TF-IDF vektörizer
vectorizer_tfidf = load_model_safe("models/tf-idf/tfidf_vectorizer.pkl")
# Word2Vec yükle
try:
    if os.path.exists("models/word2vec/word2vec_model.model"):
        w2v_model = Word2Vec.load("models/word2vec/word2vec_model.model")
        print("✅ Word2Vec modeli yüklendi")
    else:
        w2v_model = None
        print("⚠️ Word2Vec modeli bulunamadı")
except Exception as e:
    w2v_model = None
    print(f"❌ Word2Vec modeli yüklenemedi: {e}")

DATA_PATH = "data/training_data.xlsx"

# Log klasörü ve dosyası
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "user_actions.log")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_user_action(ip, action_type, haber, skorlar, model):
    logging.info(f"IP: {ip} | Action: {action_type} | Model: {model} | Haber: {haber[:100]} | Skorlar: {skorlar}")

def detect_language(text):
    try:
        lang = detect(str(text))
        if lang == 'tr':
            return 'tr'
        elif lang == 'en':
            return 'en'
        else:
            return lang  # başka bir dil ise orijinal kodu koru
    except LangDetectException:
        # fallback: eski yöntem
        turkce_karakterler = set('çğıöşüÇĞIÖŞÜ')
        if any(char in turkce_karakterler for char in str(text)):
            return 'tr'
        return 'en'

def normalize_text(text):
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'[\s\n\r]+', ' ', text)  # Tüm boşlukları tek boşluğa indir
    text = re.sub(r'[^\w\s]', '', text)      # Noktalama işaretlerini kaldır
    return text.strip()

def get_existing_normalized_set(df):
    # Eğer content_norm sütunu varsa onu kullan, yoksa content'i normalize et
    if 'content_norm' in df.columns:
        return set(df['content_norm'].dropna().astype(str))
    else:
        return set(df['content'].dropna().astype(str).apply(normalize_text))

def tokenize(text):
    return str(text).lower().split()

def get_sentence_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def get_sentence_vector_glove(tokens, glove):
    vectors = [glove[word] for word in tokens if word in glove]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(glove.vector_size)

def predict_deep_learning(text, model_name):
    """Deep Learning modeli ile tahmin yap"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    if model_name not in DL_MODELS or DL_TOKENIZER is None:
        print(f"Model kontrolü: {model_name} in DL_MODELS: {model_name in DL_MODELS}")
        print(f"Tokenizer kontrolü: DL_TOKENIZER is None: {DL_TOKENIZER is None}")
        return None
    
    try:
        # Metni tokenize et
        sequences = DL_TOKENIZER.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
        
        # Tahmin yap
        predictions = DL_MODELS[model_name].predict(padded, verbose=0)
        
        # Debug: Tahmin değerlerini yazdır
        print(f"Raw predictions for {model_name}: {predictions[0]}")
        
        # Daha agresif yuvarlama - daha çeşitli skorlar için
        rounded_preds = []
        for pred in predictions[0]:
            # 0-5 aralığına sınırla
            pred = np.clip(pred, 0, 5)
            
            # Daha agresif yuvarlama - 5 skorunu da dahil et
            if pred < 1.5:
                rounded_preds.append(0)
            elif pred < 2.5:
                rounded_preds.append(1)
            elif pred < 3.5:
                rounded_preds.append(2)
            elif pred < 4.5:
                rounded_preds.append(3)
            else:
                rounded_preds.append(5)  # 4.5+ değerler 5'e yuvarlanmalı
        
        print(f"Rounded predictions for {model_name}: {rounded_preds}")
        
        return {
            "Dolar": rounded_preds[0],
            "Altın": rounded_preds[1],
            "Borsa": rounded_preds[2],
            "Bitcoin": rounded_preds[3]
        }
    except Exception as e:
        print(f"Deep Learning tahmin hatası: {e}")
        return None



def haber_temizle(metin):
    """Haber metnini temizler - gereksiz boşlukları ve satırları kaldırır"""
    if not isinstance(metin, str):
        return str(metin)
    
    # Çoklu boşlukları tek boşluğa çevir
    metin = re.sub(r'\s+', ' ', metin)
    # Başındaki ve sonundaki boşlukları kaldır
    metin = metin.strip()
    # Çoklu satır sonlarını tek satır sonuna çevir
    metin = re.sub(r'\n\s*\n', '\n', metin)
    # Satır başındaki ve sonundaki boşlukları kaldır
    metin = re.sub(r'\n\s+', '\n', metin)
    metin = re.sub(r'\s+\n', '\n', metin)
    
    return metin

def haber_ozetle(metin, dil='tr', max_cumle=2):
    """Haber metninden özet çıkarır"""
    try:
        if not isinstance(metin, str) or len(metin) < 40:
            return metin
        cumleler = sent_tokenize(metin)
        if len(cumleler) <= max_cumle:
            return metin
        # Basit extractive: en uzun cümleleri seç
        secili = sorted(cumleler, key=lambda x: len(x), reverse=True)[:max_cumle]
        # Orijinal sıraya göre sırala
        secili = sorted(secili, key=lambda x: cumleler.index(x))
        return ' '.join(secili)
    except Exception:
        return metin[:200] + '...'

@lru_cache(maxsize=1)
def get_vectorizer_tfidf():
    try:
        if os.path.exists("models/tf-idf/tfidf_vectorizer.pkl"):
            return joblib.load("models/tf-idf/tfidf_vectorizer.pkl")
        else:
            print("⚠️ TF-IDF vectorizer dosyası bulunamadı")
            return None
    except Exception as e:
        print(f"❌ TF-IDF vectorizer yükleme hatası: {e}")
        return None

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    skorlar = None
    haber = ""
    secili_model = "rf"
    secili_yontem = "tfidf"
    secili_dl_model = "cnn"  # Varsayılan değer
    if request.method == "POST":
        haber = request.form["haber"]
        secili_yontem = request.form.get("yontem", "tfidf")
        secili_model = request.form.get("model", "rf")
        secili_dl_model = request.form.get("dl_model", "cnn")
        if len(haber.split()) < 4:
            skorlar = {"Dolar": 3, "Altın": 3, "Borsa": 3, "Bitcoin": 3}
        else:
            if secili_yontem == "tfidf":
                vectorizer_tfidf = get_vectorizer_tfidf()
                if vectorizer_tfidf is None:
                    skorlar = {"Dolar": "TF-IDF Vectorizer Yok", "Altın": "TF-IDF Vectorizer Yok", "Borsa": "TF-IDF Vectorizer Yok", "Bitcoin": "TF-IDF Vectorizer Yok"}
                    return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
                X = vectorizer_tfidf.transform([haber])
            elif secili_yontem == "w2v":
                if w2v_model is None:
                    skorlar = {"Dolar": "Word2Vec Model Yok", "Altın": "Word2Vec Model Yok", "Borsa": "Word2Vec Model Yok", "Bitcoin": "Word2Vec Model Yok"}
                    return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
                tokens = tokenize(haber)
                X = np.array([get_sentence_vector(tokens, w2v_model)])
            elif secili_yontem == "glove":
                tokens = tokenize(haber)
                if glove_vectors is not None:
                    X = np.array([get_sentence_vector_glove(tokens, glove_vectors)])
                else:
                    skorlar = {"Dolar": "GloVe yok", "Altın": "GloVe yok", "Borsa": "GloVe yok", "Bitcoin": "GloVe yok"}
                    return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
            elif secili_yontem == "deep_learning":
                # Deep Learning tahmini
                if not TENSORFLOW_AVAILABLE:
                    skorlar = {"Dolar": "TensorFlow Yok", "Altın": "TensorFlow Yok", "Borsa": "TensorFlow Yok", "Bitcoin": "TensorFlow Yok"}
                    return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
                
                dl_result = predict_deep_learning(haber, secili_dl_model)
                if dl_result is not None:
                    skorlar = dl_result
                else:
                    # DL modelleri yoksa TF-IDF ile fallback yap
                    print("DL modelleri bulunamadı, TF-IDF ile fallback yapılıyor...")
                    vectorizer_tfidf = get_vectorizer_tfidf()
                    X = vectorizer_tfidf.transform([haber])
                    modeller = model_files["tfidf"]["rf"]  # Random Forest ile fallback
                    skorlar = {
                        "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X)[0]))),
                        "Altın": min(5, max(1, round(modeller["Altın"].predict(X)[0]))),
                        "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X)[0]))),
                        "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X)[0]))),
                    }
                log_user_action(request.remote_addr, f"tahmin_{secili_yontem}_{secili_dl_model}", haber, skorlar, secili_model)
                return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)

            if secili_yontem == "tfidf" and secili_model == "nb":
                X_input = X.toarray()
            else:
                X_input = X
            
            # Model kontrolü
            if secili_yontem not in model_files or secili_model not in model_files[secili_yontem]:
                skorlar = {"Dolar": "Model Yok", "Altın": "Model Yok", "Borsa": "Model Yok", "Bitcoin": "Model Yok"}
                return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
            
            modeller = model_files[secili_yontem][secili_model]
            
            # Model dosyalarının varlığını kontrol et
            if any(model is None for model in modeller.values()):
                skorlar = {"Dolar": "Model Dosyası Yok", "Altın": "Model Dosyası Yok", "Borsa": "Model Dosyası Yok", "Bitcoin": "Model Dosyası Yok"}
                return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)
            
            skorlar = {
                "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X_input)[0]))),
                "Altın": min(5, max(1, round(modeller["Altın"].predict(X_input)[0]))),
                "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X_input)[0]))),
                "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X_input)[0]))),
            }
        log_user_action(request.remote_addr, f"tahmin_{secili_yontem}", haber, skorlar, secili_model)
    return render_template("index.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_yontem=secili_yontem, secili_dl_model=secili_dl_model)

@app.route("/ekle", methods=["GET", "POST"])
def ekle():
    skorlar = None
    haber = ""
    secili_model = "rf"
    secili_dl_model = "cnn"  # Varsayılan değer
    mesaj = ""
    kullanici_skor = {}
    if request.method == "POST":
        haber = request.form["haber"]
        secili_model = request.form.get("model", "rf")
        # Eğer kullanıcı skorları submit ettiyse onları al
        if "submit" in request.form:
            # Form field isimlerini düzelt (Türkçe karakter sorunu)
            kullanici_skor = {
                "Dolar": max(1, int(request.form.get("skor_dolar", 3))),  # Minimum 1
                "Altın": max(1, int(request.form.get("skor_altin", 3))),  # HTML'de skor_altin
                "Borsa": max(1, int(request.form.get("skor_borsa", 3))),  # Minimum 1
                "Bitcoin": max(1, int(request.form.get("skor_bitcoin", 3)))  # Minimum 1
            }
            haber_norm = normalize_text(haber)
            if os.path.exists(DATA_PATH):
                df = pd.read_excel(DATA_PATH)
                existing_norms = get_existing_normalized_set(df)
                if haber_norm in existing_norms:
                    mesaj = "Bu haber zaten eğitim setinde mevcut!"
                    skorlar = kullanici_skor
                    return render_template("ekle.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_dl_model=secili_dl_model, mesaj=mesaj)
            # Haber metnini temizle
            haber_temizlenmis = haber_temizle(haber)
            # Haber özetini çıkar
            haber_ozeti = haber_ozetle(haber_temizlenmis, detect_language(haber_temizlenmis))
            
            new_row = {
                "content": haber_temizlenmis,
                "ozet": haber_ozeti,
                "content_norm": haber_norm,
                "language": detect_language(haber_temizlenmis),
                "dolar_skor": kullanici_skor["Dolar"],
                "altin_skor": kullanici_skor["Altın"],
                "borsa_skor": kullanici_skor["Borsa"],
                "bitcoin_skor": kullanici_skor["Bitcoin"]
            }
            if os.path.exists(DATA_PATH):
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            else:
                df = pd.DataFrame([new_row])
            df.to_excel(DATA_PATH, index=False)
            mesaj = "Haber başarıyla eğitim setine eklendi!"
            skorlar = kullanici_skor  # Formda tekrar gösterilsin
            # Log ekleme işlemi
            log_user_action(request.remote_addr, "ekle", haber, kullanici_skor, secili_model)
        else:
            # Tahmin et
            if len(haber.split()) < 4:
                skorlar = {"Dolar": 3, "Altın": 3, "Borsa": 3, "Bitcoin": 3}
            else:
                vectorizer_tfidf = get_vectorizer_tfidf()
                X = vectorizer_tfidf.transform([haber]) # tfidf kullanılıyor
                modeller = model_files["tfidf"][secili_model]
                if secili_model == "nb":
                    X_input = X.toarray()
                else:
                    X_input = X
                skorlar = {
                    "Dolar": min(5, max(1, round(modeller["Dolar"].predict(X_input)[0]))),
                    "Altın": min(5, max(1, round(modeller["Altın"].predict(X_input)[0]))),
                    "Borsa": min(5, max(1, round(modeller["Borsa"].predict(X_input)[0]))),
                    "Bitcoin": min(5, max(1, round(modeller["Bitcoin"].predict(X_input)[0]))),
                }
            # Log tahmin işlemi (ekle sayfasında)
            log_user_action(request.remote_addr, "tahmin_ekle", haber, skorlar, secili_model)
    return render_template("ekle.html", skorlar=skorlar, haber=haber, secili_model=secili_model, secili_dl_model=secili_dl_model, mesaj=mesaj)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)