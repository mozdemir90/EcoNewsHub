import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os

# 1. Etiketli veriyi oku
df_train = pd.read_excel("data/training_data4.xlsx")
df_test = pd.read_excel("data/analiz_sonuclari2.xlsx")

# 2. TF-IDF vektörleştirici
# Bilgi kaybını azaltmak için 'content' + 'ozet' birleşik metni kullan
train_text = (
    df_train.get("content", "").astype(str).fillna("") + " " + df_train.get("ozet", "").astype(str).fillna("")
).str.strip()
test_text = (
    df_test.get("content", "").astype(str).fillna("") + " " + df_test.get("ozet", "").astype(str).fillna("")
).str.strip()

# Daha fazla bağlam için unigram + bigram, daha sağlam için min_df/max_df ve sublinear_tf
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
)
X_all = vectorizer.fit_transform(train_text)
X_test = vectorizer.transform(test_text)

varliklar = ["dolar_skor", "altin_skor", "borsa_skor", "bitcoin_skor"]
tahminler_rf = {}
tahminler_svm = {}
tahminler_nb = {}
tahminler_ada = {}
tahminler_ann = {}
ann_models = {}
rf_models = {}
svm_models = {}
nb_models = {}
ada_models = {}

os.makedirs("models", exist_ok=True)
os.makedirs("models/tf-idf", exist_ok=True)

for varlik in varliklar:
    y = df_train[varlik]
    # Train/Val split (20% validation)
    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y, test_size=0.2, random_state=42)
    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_tr, y_tr)
    # simple val metric
    val_mae_rf = mean_absolute_error(y_val, model_rf.predict(X_val))
    tahmin_rf = model_rf.predict(X_test)
    tahmin_rf = [min(5, max(1, round(t))) for t in tahmin_rf]
    tahminler_rf[varlik + "_rf"] = tahmin_rf
    rf_models[varlik] = model_rf
    joblib.dump(model_rf, f"models/tf-idf/{varlik}_rf_model.pkl")

    # SVM
    model_svm = SVR()
    model_svm.fit(X_tr, y_tr)
    val_mae_svm = mean_absolute_error(y_val, model_svm.predict(X_val))
    tahmin_svm = model_svm.predict(X_test)
    tahmin_svm = [min(5, max(1, round(t))) for t in tahmin_svm]
    tahminler_svm[varlik + "_svm"] = tahmin_svm
    svm_models[varlik] = model_svm
    joblib.dump(model_svm, f"models/tf-idf/{varlik}_svm_model.pkl")

    # Naive Bayes (NB için dense array gerekir)
    model_nb = GaussianNB()
    model_nb.fit(X_tr.toarray(), y_tr)
    val_mae_nb = mean_absolute_error(y_val, model_nb.predict(X_val.toarray()))
    tahmin_nb = model_nb.predict(X_test.toarray())
    tahmin_nb = [min(5, max(1, round(t))) for t in tahmin_nb]
    tahminler_nb[varlik + "_nb"] = tahmin_nb
    nb_models[varlik] = model_nb
    joblib.dump(model_nb, f"models/tf-idf/{varlik}_nb_model.pkl")

    # AdaBoost (dense array gerekir)
    model_ada = AdaBoostRegressor(n_estimators=100, random_state=42)
    model_ada.fit(X_tr.toarray(), y_tr)
    val_mae_ada = mean_absolute_error(y_val, model_ada.predict(X_val.toarray()))
    tahmin_ada = model_ada.predict(X_test.toarray())
    tahmin_ada = [min(5, max(1, round(t))) for t in tahmin_ada]
    tahminler_ada[varlik + "_ada"] = tahmin_ada
    ada_models[varlik] = model_ada
    joblib.dump(model_ada, f"models/tf-idf/{varlik}_ada_model.pkl")

    # ANN (MLPRegressor, dense array gerekir)
    model_ann = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    model_ann.fit(X_tr.toarray(), y_tr)
    val_mae_ann = mean_absolute_error(y_val, model_ann.predict(X_val.toarray()))
    tahmin_ann = model_ann.predict(X_test.toarray())
    tahmin_ann = [min(5, max(1, round(t))) for t in tahmin_ann]
    tahminler_ann[varlik + "_ann"] = tahmin_ann
    ann_models[varlik] = model_ann
    joblib.dump(model_ann, f"models/tf-idf/{varlik}_ann_model.pkl")

    print(f"[VAL] {varlik} | RF MAE={val_mae_rf:.4f} | SVM MAE={val_mae_svm:.4f} | NB MAE={val_mae_nb:.4f} | ADA MAE={val_mae_ada:.4f} | ANN MAE={val_mae_ann:.4f}")

    # 5-fold cross-validation (report MAE)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_rf_list, mae_svm_list, mae_nb_list, mae_ada_list, mae_ann_list = [], [], [], [], []
    for tr_idx, va_idx in kf.split(X_all):
        Xtr, Xva = X_all[tr_idx], X_all[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

        m_rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xtr, ytr)
        mae_rf_list.append(mean_absolute_error(yva, m_rf.predict(Xva)))

        m_svm = SVR().fit(Xtr, ytr)
        mae_svm_list.append(mean_absolute_error(yva, m_svm.predict(Xva)))

        m_nb = GaussianNB().fit(Xtr.toarray(), ytr)
        mae_nb_list.append(mean_absolute_error(yva, m_nb.predict(Xva.toarray())))

        m_ada = AdaBoostRegressor(n_estimators=100, random_state=42).fit(Xtr.toarray(), ytr)
        mae_ada_list.append(mean_absolute_error(yva, m_ada.predict(Xva.toarray())))

        m_ann = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42).fit(Xtr.toarray(), ytr)
        mae_ann_list.append(mean_absolute_error(yva, m_ann.predict(Xva.toarray())))

    print(f"[CV5] {varlik} | RF MAE={sum(mae_rf_list)/5:.4f} | SVM MAE={sum(mae_svm_list)/5:.4f} | NB MAE={sum(mae_nb_list)/5:.4f} | ADA MAE={sum(mae_ada_list)/5:.4f} | ANN MAE={sum(mae_ann_list)/5:.4f}")

# 3. Sonuçları yeni DataFrame'e ekle
for varlik in varliklar:
    df_test[varlik + "_rf"] = tahminler_rf[varlik + "_rf"]
    df_test[varlik + "_svm"] = tahminler_svm[varlik + "_svm"]
    df_test[varlik + "_nb"] = tahminler_nb[varlik + "_nb"]
    df_test[varlik + "_ada"] = tahminler_ada[varlik + "_ada"]
    df_test[varlik + "_ann"] = tahminler_ann[varlik + "_ann"]

# 4. Eğitim seti güncelleme işlemi overfitting nedeniyle durduruldu
# Test verilerinin eğitim setine eklenmesi overfittinge sebep oluyor
print("Not: Test verilerinin eğitim setine eklenmesi overfitting nedeniyle durduruldu.")

# 5. Tahminli sonuçları kaydet
df_test.to_excel("data/analiz_sonuclari2_tahminli_TF-IDF.xlsx", index=False)
print("Tahminler data/analiz_sonuclari2_tahminli_TF-IDF.xlsx dosyasına kaydedildi.")

# TF-IDF vectorizer'ı da kaydet
joblib.dump(vectorizer, "models/tf-idf/tfidf_vectorizer.pkl")
