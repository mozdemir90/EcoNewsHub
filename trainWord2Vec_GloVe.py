import pandas as pd
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Etiketli veriyi oku
df_train = pd.read_excel("data/training_data4.xlsx")
df_test = pd.read_excel("data/analiz_sonuclari2.xlsx")
varliklar = ["dolar_skor", "altin_skor", "borsa_skor", "bitcoin_skor"]

# 2. Tokenizasyon fonksiyonu
def tokenize(text):
    return str(text).lower().split()

# 3. Word2Vec modeli eğit (veya hazır model yükle)
# Bilgi kaybını azaltmak için 'content' + 'ozet' birleşik metni ile eğit
sentences = (
    df_train.get("content", "").astype(str).fillna("") + " " + df_train.get("ozet", "").astype(str).fillna("")
).apply(tokenize).tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
# Alternatif: w2v_model = KeyedVectors.load_word2vec_format('PATH/GoogleNews-vectors-negative300.bin', binary=True)

# 4. GloVe vektörlerini yükle
GLOVE_PATH = "data/glove.6B.100d.txt"
glove_vectors = None
if os.path.exists(GLOVE_PATH):
    glove_vectors = KeyedVectors.load_word2vec_format(GLOVE_PATH, binary=False, no_header=True)
else:
    print(f"UYARI: {GLOVE_PATH} bulunamadı, GloVe ile eğitim yapılmayacak.")

# 5. Cümle vektörü çıkarma fonksiyonları
def get_sentence_vector_w2v(tokens, model):
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

# 6. Eğitim ve test için vektör matrisleri oluştur (Word2Vec) - birleşik metin
train_texts = (df_train.get("content", "").astype(str).fillna("") + " " + df_train.get("ozet", "").astype(str).fillna("")).tolist()
test_texts = (df_test.get("content", "").astype(str).fillna("") + " " + df_test.get("ozet", "").astype(str).fillna("")).tolist()
X_all_w2v = np.vstack([get_sentence_vector_w2v(tokenize(text), w2v_model) for text in train_texts])
X_test_w2v = np.vstack([get_sentence_vector_w2v(tokenize(text), w2v_model) for text in test_texts])

# 6b. Eğitim ve test için vektör matrisleri oluştur (GloVe) - birleşik metin
if glove_vectors is not None:
    X_all_glove = np.vstack([get_sentence_vector_glove(tokenize(text), glove_vectors) for text in train_texts])
    X_test_glove = np.vstack([get_sentence_vector_glove(tokenize(text), glove_vectors) for text in test_texts])

# 7. Her varlık için model eğit ve tahmin et (Word2Vec)
os.makedirs("models/word2vec", exist_ok=True)
for varlik in varliklar:
    y = df_train[varlik]
    # Split for validation (20%)
    X_tr_w2v, X_val_w2v, y_tr, y_val = train_test_split(X_all_w2v, y, test_size=0.2, random_state=42)
    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_tr_w2v, y_tr)
    print(f"[VAL][w2v] {varlik} RF MAE={mean_absolute_error(y_val, model_rf.predict(X_val_w2v)):.4f}")
    df_test[varlik + "_rf_w2v"] = [min(5, max(1, round(t))) for t in model_rf.predict(X_test_w2v)]
    joblib.dump(model_rf, f"models/word2vec/{varlik}_rf_w2v_model.pkl")
    # SVM
    model_svm = SVR()
    model_svm.fit(X_tr_w2v, y_tr)
    print(f"[VAL][w2v] {varlik} SVM MAE={mean_absolute_error(y_val, model_svm.predict(X_val_w2v)):.4f}")
    df_test[varlik + "_svm_w2v"] = [min(5, max(1, round(t))) for t in model_svm.predict(X_test_w2v)]
    joblib.dump(model_svm, f"models/word2vec/{varlik}_svm_w2v_model.pkl")
    # ANN
    model_ann = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    model_ann.fit(X_tr_w2v, y_tr)
    print(f"[VAL][w2v] {varlik} ANN MAE={mean_absolute_error(y_val, model_ann.predict(X_val_w2v)):.4f}")
    df_test[varlik + "_ann_w2v"] = [min(5, max(1, round(t))) for t in model_ann.predict(X_test_w2v)]
    joblib.dump(model_ann, f"models/word2vec/{varlik}_ann_w2v_model.pkl")
    # ADA
    model_ada = AdaBoostRegressor(n_estimators=100, random_state=42)
    model_ada.fit(X_tr_w2v, y_tr)
    print(f"[VAL][w2v] {varlik} ADA MAE={mean_absolute_error(y_val, model_ada.predict(X_val_w2v)):.4f}")
    df_test[varlik + "_ada_w2v"] = [min(5, max(1, round(t))) for t in model_ada.predict(X_test_w2v)]

    # 5-fold CV for W2V
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = {'rf': [], 'svm': [], 'ann': [], 'ada': []}
    for tr, va in kf.split(X_all_w2v):
        Xtr, Xva = X_all_w2v[tr], X_all_w2v[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        cv_mae['rf'].append(mean_absolute_error(yva, RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xva)))
        cv_mae['svm'].append(mean_absolute_error(yva, SVR().fit(Xtr, ytr).predict(Xva)))
        cv_mae['ann'].append(mean_absolute_error(yva, MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42).fit(Xtr, ytr).predict(Xva)))
        cv_mae['ada'].append(mean_absolute_error(yva, AdaBoostRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xva)))
    print(f"[CV5][w2v] {varlik} | RF={sum(cv_mae['rf'])/5:.4f} | SVM={sum(cv_mae['svm'])/5:.4f} | ANN={sum(cv_mae['ann'])/5:.4f} | ADA={sum(cv_mae['ada'])/5:.4f}")
    joblib.dump(model_ada, f"models/word2vec/{varlik}_ada_w2v_model.pkl")

# 7b. Her varlık için model eğit ve tahmin et (GloVe)
if glove_vectors is not None:
    for varlik in varliklar:
        y = df_train[varlik]
        X_tr_gl, X_val_gl, y_tr, y_val = train_test_split(X_all_glove, y, test_size=0.2, random_state=42)
        # Random Forest
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X_tr_gl, y_tr)
        print(f"[VAL][glove] {varlik} RF MAE={mean_absolute_error(y_val, model_rf.predict(X_val_gl)):.4f}")
        df_test[varlik + "_rf_glove"] = [min(5, max(1, round(t))) for t in model_rf.predict(X_test_glove)]
        joblib.dump(model_rf, f"models/glove/{varlik}_rf_glove_model.pkl")
        # SVM
        model_svm = SVR()
        model_svm.fit(X_tr_gl, y_tr)
        print(f"[VAL][glove] {varlik} SVM MAE={mean_absolute_error(y_val, model_svm.predict(X_val_gl)):.4f}")
        df_test[varlik + "_svm_glove"] = [min(5, max(1, round(t))) for t in model_svm.predict(X_test_glove)]
        joblib.dump(model_svm, f"models/glove/{varlik}_svm_glove_model.pkl")
        # ANN
        model_ann = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        model_ann.fit(X_tr_gl, y_tr)
        print(f"[VAL][glove] {varlik} ANN MAE={mean_absolute_error(y_val, model_ann.predict(X_val_gl)):.4f}")
        df_test[varlik + "_ann_glove"] = [min(5, max(1, round(t))) for t in model_ann.predict(X_test_glove)]
        joblib.dump(model_ann, f"models/glove/{varlik}_ann_glove_model.pkl")
        # ADA
        model_ada = AdaBoostRegressor(n_estimators=100, random_state=42)
        model_ada.fit(X_tr_gl, y_tr)
        print(f"[VAL][glove] {varlik} ADA MAE={mean_absolute_error(y_val, model_ada.predict(X_val_gl)):.4f}")
        df_test[varlik + "_ada_glove"] = [min(5, max(1, round(t))) for t in model_ada.predict(X_test_glove)]

        # 5-fold CV for GloVe
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = {'rf': [], 'svm': [], 'ann': [], 'ada': []}
        for tr, va in kf.split(X_all_glove):
            Xtr, Xva = X_all_glove[tr], X_all_glove[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            cv_mae['rf'].append(mean_absolute_error(yva, RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xva)))
            cv_mae['svm'].append(mean_absolute_error(yva, SVR().fit(Xtr, ytr).predict(Xva)))
            cv_mae['ann'].append(mean_absolute_error(yva, MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42).fit(Xtr, ytr).predict(Xva)))
            cv_mae['ada'].append(mean_absolute_error(yva, AdaBoostRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr).predict(Xva)))
        print(f"[CV5][glove] {varlik} | RF={sum(cv_mae['rf'])/5:.4f} | SVM={sum(cv_mae['svm'])/5:.4f} | ANN={sum(cv_mae['ann'])/5:.4f} | ADA={sum(cv_mae['ada'])/5:.4f}")
        joblib.dump(model_ada, f"models/glove/{varlik}_ada_glove_model.pkl")

# 8. Sonuçları kaydet
df_test.to_excel("data/analiz_sonuclari2_tahminli_w2v.xlsx", index=False)
if glove_vectors is not None:
    df_test.to_excel("data/analiz_sonuclari2_tahminli_glove.xlsx", index=False)
if hasattr(w2v_model, 'save'):
    w2v_model.save("models/word2vec/word2vec_model.model")
print("Word2Vec ve (varsa) GloVe ile tahminler kaydedildi.")

# --- Eğitim seti güncelleme işlemi overfitting nedeniyle durduruldu ---
print("Not: Test verilerinin eğitim setine eklenmesi overfitting nedeniyle durduruldu.") 
    