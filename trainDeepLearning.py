import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Bidirectional, Embedding, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# GPU kullanımını optimize et
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU kullanılıyor")
else:
    print("CPU kullanılıyor")

class DeepLearningTrainer:
    def __init__(self, max_words=10000, max_len=200):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None
        self.models = {}
        self.histories = {}
        
    def load_data(self):
        """Eğitim verisini yükle"""
        print("📥 Veri yükleniyor...")
        self.df_train = pd.read_excel("data/training_data.xlsx")
        self.df_test = pd.read_excel("data/analiz_sonuclari2.xlsx")
        
        # Boş değerleri temizle
        self.df_train = self.df_train.dropna(subset=['ozet'])
        self.df_test = self.df_test.dropna(subset=['ozet'])
        
        print(f"Eğitim verisi: {len(self.df_train)} satır")
        print(f"Test verisi: {len(self.df_test)} satır")
        
    def prepare_text_data(self):
        """Metin verilerini hazırla"""
        print("🔤 Metin verileri hazırlanıyor...")
        
        # Tokenizer oluştur ve eğit
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df_train['ozet'].astype(str))
        
        # Eğitim verisi
        X_train = self.tokenizer.texts_to_sequences(self.df_train['ozet'].astype(str))
        X_train = pad_sequences(X_train, maxlen=self.max_len, padding='post', truncating='post')
        
        # Test verisi
        X_test = self.tokenizer.texts_to_sequences(self.df_test['ozet'].astype(str))
        X_test = pad_sequences(X_test, maxlen=self.max_len, padding='post', truncating='post')
        
        # Hedef değişkenler
        varliklar = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        self.y_train = self.df_train[varliklar].values
        self.y_test = self.df_test[varliklar].values
        
        self.X_train = X_train
        self.X_test = X_test
        
        print(f"Eğitim şekli: {X_train.shape}")
        print(f"Test şekli: {X_test.shape}")
        
        # Tokenizer'ı kaydet
        joblib.dump(self.tokenizer, "models/deep_learning/tokenizer.pkl")
        
    def create_cnn_model(self):
        """1D CNN modeli oluştur"""
        print("🏗️ 1D CNN modeli oluşturuluyor...")
        
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='linear')  # 4 varlık için
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_lstm_model(self):
        """LSTM modeli oluştur"""
        print("🏗️ LSTM modeli oluşturuluyor...")
        
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_cnn_lstm_model(self):
        """CNN + LSTM hibrit modeli oluştur"""
        print("🏗️ CNN + LSTM hibrit modeli oluşturuluyor...")
        
        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Conv1D(128, 5, activation='relu'),
            MaxPooling1D(5),
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(4, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, model, model_name, epochs=50, batch_size=32):
        """Modeli eğit"""
        print(f"🎯 {model_name} eğitiliyor...")
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f"models/deep_learning/{model_name}_best.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # Eğitim
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        self.models[model_name] = model
        self.histories[model_name] = history
        
        return model, history
    
    def evaluate_model(self, model, model_name):
        """Model performansını değerlendir"""
        print(f"📊 {model_name} değerlendiriliyor...")
        
        # Tahminler
        y_pred = model.predict(self.X_test)
        
        # Metrikler
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print(f"{model_name} Sonuçları:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Test setine tahminleri ekle
        varliklar = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        for i, varlik in enumerate(varliklar):
            self.df_test[f'{varlik}_{model_name}'] = np.clip(y_pred[:, i], 0, 5).round()
        
        return mse, mae, r2
    
    def train_all_models(self):
        """Tüm modelleri eğit"""
        print("🚀 Deep Learning modelleri eğitiliyor...")
        
        # Model oluşturma
        models = {
            'cnn': self.create_cnn_model(),
            'lstm': self.create_lstm_model(),
            'cnn_lstm': self.create_cnn_lstm_model()
        }
        
        # Eğitim ve değerlendirme
        results = {}
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"🎯 {model_name.upper()} MODELİ")
            print(f"{'='*50}")
            
            # Eğit
            model, history = self.train_model(model, model_name)
            
            # Değerlendir
            mse, mae, r2 = self.evaluate_model(model, model_name)
            results[model_name] = {'mse': mse, 'mae': mae, 'r2': r2}
            
            # Modeli kaydet
            model.save(f"models/deep_learning/{model_name}_model.h5")
        
        return results
    
    def save_results(self):
        """Sonuçları kaydet"""
        print("💾 Sonuçlar kaydediliyor...")
        
        # Test sonuçlarını kaydet
        self.df_test.to_excel("data/analiz_sonuclari2_tahminli_DL.xlsx", index=False)
        
        # Performans özeti
        results_df = pd.DataFrame()
        for model_name in self.models.keys():
            mse, mae, r2 = self.evaluate_model(self.models[model_name], model_name)
            results_df = pd.concat([results_df, pd.DataFrame({
                'Model': [model_name],
                'MSE': [mse],
                'MAE': [mae],
                'R²': [r2]
            })], ignore_index=True)
        
        results_df.to_excel("data/deep_learning_performance.xlsx", index=False)
        print("✅ Sonuçlar kaydedildi!")
        
    def run_training(self):
        """Tüm eğitim sürecini çalıştır"""
        print("🎯 Deep Learning Eğitim Süreci Başlıyor...")
        print("="*60)
        
        # 1. Veri yükle
        self.load_data()
        
        # 2. Metin verilerini hazırla
        self.prepare_text_data()
        
        # 3. Modelleri eğit
        results = self.train_all_models()
        
        # 4. Sonuçları kaydet
        self.save_results()
        
        print("\n🎉 Deep Learning eğitimi tamamlandı!")
        print("📁 Sonuçlar:")
        print("- models/deep_learning/ klasöründe modeller")
        print("- data/analiz_sonuclari2_tahminli_DL.xlsx test sonuçları")
        print("- data/deep_learning_performance.xlsx performans özeti")

if __name__ == "__main__":
    # Klasörleri oluştur
    os.makedirs("models/deep_learning", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Eğitimi başlat
    trainer = DeepLearningTrainer()
    trainer.run_training() 