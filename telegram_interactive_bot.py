#!/usr/bin/env python3
"""
Interactive Telegram Bot for Sentiment Analysis
"""

import os
import requests
import json
import time
import logging
import joblib
from datetime import datetime

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/interactive_bot.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class InteractiveNewsBot:
    def __init__(self, bot_token):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        
        # User states for interactive scoring
        self.user_states = {}  # {chat_id: {'state': 'waiting_for_scores', 'news_text': '...', 'scores': {}}}
        
        # Model holders
        self.glove_vectors = None
        self.models = {}
        
        # Load GloVe + RF models
        self.load_models()
    
    def load_models(self):
        """Load GloVe embeddings and RF models for each asset"""
        try:
            logging.info("🔄 Loading GloVe + Random Forest models...")
            
            # Load GloVe vectors
            from gensim.models import KeyedVectors
            glove_path = 'data/glove.6B.100d.txt'
            if os.path.exists(glove_path):
                from datetime import datetime as _dt
                logging.info("📥 Loading GloVe vectors... this may take a while")
                self.glove_vectors = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
                logging.info("✅ GloVe vectors loaded")
            else:
                logging.error(f"❌ GloVe file not found: {glove_path}")
                return
            
            # Load RF models for each asset (trained on GloVe features)
            import joblib
            assets = ['dolar', 'altin', 'borsa', 'bitcoin']
            loaded = []
            for asset in assets:
                model_path = f'models/glove/{asset}_skor_rf_glove_model.pkl'
                if os.path.exists(model_path):
                    self.models[asset] = joblib.load(model_path)
                    loaded.append(asset)
                else:
                    logging.warning(f"⚠️ RF model not found for {asset}: {model_path}")
            if loaded:
                logging.info(f"✅ Loaded RF models for assets: {loaded}")
            else:
                logging.error("❌ No RF models loaded. Aborting.")
                return
            
        except Exception as e:
            logging.error(f"❌ Error loading GloVe/RF models: {e}")
    
    def preprocess_text(self, text):
        """Preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Basic preprocessing
        text = str(text).lower()
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra whitespace
        
        return text
    
    def predict_sentiment(self, text):
        """Predict sentiment using GloVe average embeddings + RF models"""
        try:
            logging.info(f"🔍 Starting GloVe+RF prediction for text: {text[:50]}...")
            
            if self.glove_vectors is None or not self.models:
                logging.error("❌ Models not loaded")
                return None
            
            # Preprocess
            processed_text = self.preprocess_text(text)
            if not processed_text:
                logging.error("❌ Text preprocessing failed")
                return None
            
            tokens = processed_text.split()
            # Average GloVe vector
            import numpy as np
            vectors = [self.glove_vectors[word] for word in tokens if word in self.glove_vectors]
            if vectors:
                X = np.mean(vectors, axis=0).reshape(1, -1)
            else:
                X = np.zeros((1, self.glove_vectors.vector_size))
            
            # Predict per asset
            predictions = {}
            name_map = {'dolar': 'USD', 'altin': 'Gold', 'borsa': 'Stock Market', 'bitcoin': 'Bitcoin'}
            for asset, model in self.models.items():
                try:
                    pred = model.predict(X)[0]
                    pred = int(min(5, max(1, round(pred))))
                    predictions[name_map[asset]] = pred
                    logging.info(f"✅ {asset} GloVe+RF prediction: {pred}")
                except Exception as e:
                    logging.error(f"❌ Error predicting {asset}: {e}")
                    predictions[name_map[asset]] = 3
            
            logging.info(f"✅ Final GloVe+RF predictions: {predictions}")
            return predictions
        except Exception as e:
            logging.error(f"❌ GloVe+RF prediction error: {e}")
            return None
    
    def get_glove_embeddings(self, text):
        """Get Glove embeddings for text"""
        try:
            import numpy as np
            
            # Split text into words
            words = text.split()
            logging.info(f"🔍 Words in text: {words}")
            
            # Get embeddings for each word
            embeddings = []
            found_words = []
            for word in words:
                try:
                    if word in self.word2vec_model.wv:
                        embeddings.append(self.word2vec_model.wv[word])
                        found_words.append(word)
                except:
                    continue
            
            logging.info(f"✅ Found words in Word2Vec: {found_words}")
            logging.info(f"✅ Total embeddings found: {len(embeddings)}")
            
            if not embeddings:
                # If no words found, use zero vector
                embedding_size = self.word2vec_model.vector_size
                logging.warning(f"⚠️ No words found in Word2Vec, using zero vector of size {embedding_size}")
                return np.zeros(embedding_size)
            
            # Average all word embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            logging.info(f"✅ Average embedding shape: {avg_embedding.shape}")
            return avg_embedding
            
        except Exception as e:
            logging.error(f"❌ Error getting Glove embeddings: {e}")
            return None
    
    def create_sentiment_message(self, text, predictions):
        """Create formatted sentiment analysis message"""
        try:
            # Asset-specific emoji mappings for sentiment scores
            asset_emojis = {
                'USD': {
                    1: "🔴",  # Very negative - Dolar düşüş
                    2: "🔴",  # Negative - Dolar düşüş
                    3: "🟡",  # Neutral - Dolar nötr
                    4: "🟢",  # Positive - Dolar yükseliş
                    5: "🟢"   # Very positive - Dolar yükseliş
                },
                'Gold': {
                    1: "🔴",  # Very negative - Altın düşüş
                    2: "🔴",  # Negative - Altın düşüş
                    3: "🟡",  # Neutral - Altın nötr
                    4: "🟢",  # Positive - Altın yükseliş
                    5: "🟢"   # Very positive - Altın yükseliş
                },
                'Stock Market': {
                    1: "🔴",  # Very negative - Borsa düşüş
                    2: "🔴",  # Negative - Borsa düşüş
                    3: "🟡",  # Neutral - Borsa nötr
                    4: "🟢",  # Positive - Borsa yükseliş
                    5: "🟢"   # Very positive - Borsa yükseliş
                },
                'Bitcoin': {
                    1: "🔴",  # Very negative - Bitcoin düşüş
                    2: "🔴",  # Negative - Bitcoin düşüş
                    3: "🟡",  # Neutral - Bitcoin nötr
                    4: "🟢",  # Positive - Bitcoin yükseliş
                    5: "🟢"   # Very positive - Bitcoin yükseliş
                }
            }
            
            # Format predictions with asset-specific colors (no symbols)
            pred_text = ""
            for asset, score in predictions.items():
                emoji = asset_emojis.get(asset, {}).get(score, "⚪")
                pred_text += f"{emoji} {asset}: {score}/5\n"
            
            # Create message
            message = f"""
📊 **Analysis Results **

📝 **Text:** {text[:100]}{'...' if len(text) > 100 else ''}

🎯 **Scores:**
{pred_text}

🤖 **Model:** GloVe + Random Forest
⏰ **Analyzed at:** {datetime.now().strftime('%H:%M:%S')}
            """.strip()
            
            return message
            
        except Exception as e:
            logging.error(f"❌ Error creating message: {e}")
            return None
    
    def send_message(self, chat_id, message):
        """Send message to Telegram chat"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                logging.info(f"✅ Message sent to chat {chat_id}")
                return True
            else:
                logging.error(f"❌ Failed to send message: {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error sending message: {e}")
            return False
    
    def get_updates(self):
        """Get updates from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 30
            }
            
            response = requests.get(url, params=params, timeout=35)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('ok'):
                    return data.get('result', [])
            
            return []
            
        except Exception as e:
            logging.error(f"❌ Error getting updates: {e}")
            return []
    
    def process_message(self, message):
        """Process incoming message"""
        try:
            chat_id = message.get('chat', {}).get('id')
            text = message.get('text', '')
            user_name = message.get('from', {}).get('first_name', 'User')
            
            if not text:
                return
            
            logging.info(f"📨 Received message from {user_name}: {text[:50]}...")
            
            # Check for help command
            if text.startswith('/yardim') or text.startswith('/help'):
                self.send_help_message(chat_id)
                return
            
            # Check for interactive scoring state
            if chat_id in self.user_states and self.user_states[chat_id]['state'] == 'waiting_for_scores':
                self.handle_score_input(chat_id, text, user_name)
                return
            
            # Check for /ekle command
            if text.startswith('/ekle'):
                self.handle_add_command(chat_id, text, user_name)
                return
            
            # Check for /ekle_interactive command
            if text.startswith('/ekle_interactive'):
                self.start_interactive_scoring(chat_id, text, user_name)
                return
            
            # Get sentiment prediction (GloVe + RF)
            predictions = self.predict_sentiment(text)
            
            if predictions:
                # Create response message
                response_message = self.create_sentiment_message(text, predictions)
                
                if response_message:
                    # Send response
                    self.send_message(chat_id, response_message)
                else:
                    self.send_message(chat_id, "❌ Error creating analysis message")
            else:
                self.send_message(chat_id, "❌ Error analyzing sentiment")
            
        except Exception as e:
            logging.error(f"❌ Error processing message: {e}")
    
    def send_help_message(self, chat_id):
        """Send help message with bot usage instructions"""
        help_message = """
🤖 *Interactive News Bot - Yardım*

Bu bot, ekonomi haberlerini analiz ederek farklı varlıklar için etki analizi yapar.

📝 *Komutlar:*

🔍 *Otomatik Analiz*
Herhangi bir ekonomi haberi gönderin, bot otomatik olarak analiz eder.

📊 */ekle* - Tek Seferde Haber Ekleme
`/ekle <haber_metni> | dolar:3 altin:4 borsa:2 bitcoin:5`

*Örnek:*
`/ekle Dolar kuru yükseldi, piyasalar karışık | dolar:4 altin:3 borsa:2 bitcoin:3`

🎯 */ekle_interactive* - Etkileşimli Haber Ekleme
`/ekle_interactive <haber_metni>`

Sonrasında her varlık için ayrı ayrı skor girmeniz istenecek:

• Dolar (USD)
• Altın (Gold)
• Borsa (Stock Market)
• Bitcoin

❓ */yardim* veya */help* - Bu yardım mesajını gösterir

🎯 *Skor Sistemi:*

1️⃣ Çok Negatif - Varlık için çok kötü haber
2️⃣ Negatif - Varlık için kötü haber  
3️⃣ Nötr - Varlık için nötr haber
4️⃣ Pozitif - Varlık için iyi haber
5️⃣ Çok Pozitif - Varlık için çok iyi haber

💰 *Varlık Analizi:*

🔴 1-2: Varlık düşüş trendi
🟡 3: Varlık nötr
🟢 4-5: Varlık yükseliş trendi

🤖 *Model Bilgisi:*
• Algoritma: GloVe + Random Forest
• Dil: Türkçe
• Güncelleme: Gerçek zamanlı

⏰ *Otomatik Çalışma:*
• Cron job ile 10 dakikada bir çalışır
• Son dakika haberlerini otomatik kontrol eder
• Yeni haberleri analiz edip yorumlar

📊 *Veri Kaydetme:*
Eklenen haberler hem JSON hem Excel formatında kaydedilir ve gelecekteki model eğitimlerinde kullanılır.

💡 *İpucu:* Haber metni ne kadar detaylı olursa, analiz o kadar doğru olur!
        """.strip()
        
        self.send_message(chat_id, help_message)
    
    def handle_add_command(self, chat_id, text, user_name):
        """Handle /ekle command to add news to training dataset"""
        try:
            # Parse command: /ekle <haber_metni> | dolar:3 altin:4 borsa:2 bitcoin:5
            parts = text.split('|', 1)
            
            if len(parts) != 2:
                help_message = """
📝 **/ekle Komutu Kullanımı:**

```
/ekle <haber_metni> | dolar:3 altin:4 borsa:2 bitcoin:5
```

**Örnek:**
```
/ekle Dolar kuru yükseldi, piyasalar karışık | dolar:4 altin:3 borsa:2 bitcoin:3
```

**Skorlar:**
- 1: Çok negatif
- 2: Negatif  
- 3: Nötr
- 4: Pozitif
- 5: Çok pozitif

**Varlıklar:**
- Dolar (USD)
- Altın (Gold)
- Borsa (Stock Market)
- Bitcoin
                """.strip()
                self.send_message(chat_id, help_message)
                return
            
            news_text = parts[0].replace('/ekle', '').strip()
            scores_text = parts[1].strip()
            
            # Parse scores
            scores = {}
            score_parts = scores_text.split()
            for part in score_parts:
                if ':' in part:
                    asset, score = part.split(':', 1)
                    try:
                        score = int(score)
                        if 1 <= score <= 5:
                            scores[asset.lower()] = score
                    except ValueError:
                        pass
            
            # Validate scores
            required_assets = ['dolar', 'altin', 'borsa', 'bitcoin']
            if not all(asset in scores for asset in required_assets):
                error_msg = "❌ Tüm varlıklar için skor gerekli: dolar, altin, borsa, bitcoin"
                self.send_message(chat_id, error_msg)
                return
            
            # Add to training dataset (both JSON and Excel)
            success_json = self.add_to_training_dataset(news_text, scores, user_name)
            success_excel = self.add_to_excel_dataset(news_text, scores, user_name)
            
            if success_json and success_excel:
                success_msg = f"""
✅ **Haber Başarıyla Eklendi!**

📝 **Haber:** {news_text[:100]}{'...' if len(news_text) > 100 else ''}

🎯 **Skorlar:**
🔴 Dolar: {scores['dolar']}/5
🟡 Altın: {scores['altin']}/5  
🟢 Borsa: {scores['borsa']}/5
🟢 Bitcoin: {scores['bitcoin']}/5

👤 **Ekleyen:** {user_name}
⏰ **Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

💾 **Kaydedildi:** JSON + Excel
                """.strip()
                self.send_message(chat_id, success_msg)
            else:
                self.send_message(chat_id, "⚠️ Haber eklenirken kısmi hata oluştu")
            
        except Exception as e:
            logging.error(f"❌ Error handling add command: {e}")
            self.send_message(chat_id, "❌ Komut işlenirken hata oluştu")
    
    def add_to_training_dataset(self, news_text, scores, user_name):
        """Add news to training dataset"""
        try:
            # Create training data entry
            training_entry = {
                'text': news_text,
                'dolar_skor': scores['dolar'],
                'altin_skor': scores['altin'],
                'borsa_skor': scores['borsa'],
                'bitcoin_skor': scores['bitcoin'],
                'added_by': user_name,
                'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Load existing training data
            training_file = 'data/training_data_telegram.json'
            os.makedirs('data', exist_ok=True)
            
            existing_data = []
            if os.path.exists(training_file):
                try:
                    with open(training_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            # Add new entry
            existing_data.append(training_entry)
            
            # Save updated data
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"✅ Added training data: {news_text[:50]}... by {user_name}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error adding to training dataset: {e}")
            return False
    
    def start_interactive_scoring(self, chat_id, text, user_name):
        """Start interactive scoring process"""
        try:
            # Extract news text from command
            news_text = text.replace('/ekle_interactive', '').strip()
            
            if not news_text:
                help_msg = """
📝 **/ekle_interactive Komutu Kullanımı:**

```
/ekle_interactive <haber_metni>
```

**Örnek:**
```
/ekle_interactive Dolar kuru yükseldi, piyasalar karışık
```

Sonrasında her varlık için ayrı ayrı skor girmeniz istenecek:

- Dolar (USD)
- Altın (Gold)
- Borsa (Stock Market)
- Bitcoin
                """.strip()
                self.send_message(chat_id, help_msg)
                return
            
            # Initialize user state
            self.user_states[chat_id] = {
                'state': 'waiting_for_scores',
                'news_text': news_text,
                'scores': {},
                'user_name': user_name
            }
            
            # Send first scoring prompt
            self.send_scoring_prompt(chat_id)
            
        except Exception as e:
            logging.error(f"❌ Error starting interactive scoring: {e}")
            self.send_message(chat_id, "❌ Hata oluştu")
    
    def send_scoring_prompt(self, chat_id):
        """Send scoring prompt for next asset"""
        try:
            assets = ['dolar', 'altin', 'borsa', 'bitcoin']
            asset_names = {
                'dolar': 'Dolar (USD)',
                'altin': 'Altın (Gold)', 
                'borsa': 'Borsa (Stock Market)',
                'bitcoin': 'Bitcoin'
            }
            
            current_scores = self.user_states[chat_id]['scores']
            
            # Find next asset to score
            for asset in assets:
                if asset not in current_scores:
                    prompt = f"""
📊 **Skor Girme: {asset_names[asset]}**

📝 **Haber:** {self.user_states[chat_id]['news_text'][:100]}{'...' if len(self.user_states[chat_id]['news_text']) > 100 else ''}

🎯 **{asset_names[asset]} için skor girin (1-5):**

1️⃣ Çok negatif
2️⃣ Negatif  
3️⃣ Nötr
4️⃣ Pozitif
5️⃣ Çok pozitif

**Sadece rakam yazın (1, 2, 3, 4, 5)**
                    """.strip()
                    self.send_message(chat_id, prompt)
                    return
            
            # All scores collected, save the data
            self.save_interactive_data(chat_id)
            
        except Exception as e:
            logging.error(f"❌ Error sending scoring prompt: {e}")
    
    def handle_score_input(self, chat_id, text, user_name):
        """Handle score input from user"""
        try:
            # Parse score
            try:
                score = int(text.strip())
                if score < 1 or score > 5:
                    raise ValueError("Score out of range")
            except ValueError:
                self.send_message(chat_id, "❌ Lütfen 1-5 arası bir rakam girin")
                return
            
            # Find current asset
            assets = ['dolar', 'altin', 'borsa', 'bitcoin']
            current_scores = self.user_states[chat_id]['scores']
            
            for asset in assets:
                if asset not in current_scores:
                    current_scores[asset] = score
                    break
            
            # Send confirmation
            asset_names = {
                'dolar': 'Dolar (USD)',
                'altin': 'Altın (Gold)', 
                'borsa': 'Borsa (Stock Market)',
                'bitcoin': 'Bitcoin'
            }
            self.send_message(chat_id, f"✅ {asset_names[asset]}: {score}/5 kaydedildi")
            
            # Send next prompt or save
            self.send_scoring_prompt(chat_id)
            
        except Exception as e:
            logging.error(f"❌ Error handling score input: {e}")
            self.send_message(chat_id, "❌ Hata oluştu")
    
    def save_interactive_data(self, chat_id):
        """Save data from interactive scoring"""
        try:
            user_state = self.user_states[chat_id]
            news_text = user_state['news_text']
            scores = user_state['scores']
            user_name = user_state['user_name']
            
            # Save to JSON
            success_json = self.add_to_training_dataset(news_text, scores, user_name)
            
            # Save to Excel
            success_excel = self.add_to_excel_dataset(news_text, scores, user_name)
            
            # Send success message
            if success_json and success_excel:
                success_msg = f"""
✅ **Haber Başarıyla Eklendi!**

📝 **Haber:** {news_text[:100]}{'...' if len(news_text) > 100 else ''}

🎯 **Skorlar:**
🔴 Dolar: {scores['dolar']}/5
🟡 Altın: {scores['altin']}/5  
🟢 Borsa: {scores['borsa']}/5
🟢 Bitcoin: {scores['bitcoin']}/5

👤 **Ekleyen:** {user_name}
⏰ **Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

💾 **Kaydedildi:** JSON + Excel
                """.strip()
                self.send_message(chat_id, success_msg)
            else:
                self.send_message(chat_id, "⚠️ Veri kaydedilirken kısmi hata oluştu")
            
            # Clear user state
            del self.user_states[chat_id]
            
        except Exception as e:
            logging.error(f"❌ Error saving interactive data: {e}")
            self.send_message(chat_id, "❌ Veri kaydedilirken hata oluştu")
    
    def detect_language(self, text):
        """Detect language based on Turkish characters"""
        if not text:
            return 'tr'  # Default to Turkish
        
        text = str(text)
        # Turkish specific characters
        turkish_chars = set('çğıöşüÇĞIÖŞÜ')
        
        # Count Turkish characters
        turkish_count = sum(1 for char in text if char in turkish_chars)
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'tr'  # Default if no alphabetic characters
        
        # If more than 5% Turkish characters, consider it Turkish
        if turkish_count / total_chars > 0.05:
            return 'tr'
        else:
            return 'en'

    def add_to_excel_dataset(self, news_text, scores, user_name):
        """Add news to Excel training dataset"""
        try:
            import pandas as pd
            from openpyxl import load_workbook
            
            excel_file = 'data/training_data4.xlsx'
            
            # Detect language
            detected_language = self.detect_language(news_text)
            
            # Create new entry with all required fields (no 'text' column)
            new_entry = {
                'content': news_text,  # Ana haber metni
                'ozet': news_text,     # Özet (aynı metin)
                'language': detected_language,  # Auto-detected language
                'dolar_skor': scores['dolar'],
                'altin_skor': scores['altin'],
                'borsa_skor': scores['borsa'],
                'bitcoin_skor': scores['bitcoin'],
                'content_norm': news_text,  # Normalize edilmiş içerik
                'added_by': user_name,
                'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Load existing data
            try:
                df = pd.read_excel(excel_file)
            except:
                # Create new DataFrame with correct columns if file doesn't exist (no 'text')
                df = pd.DataFrame(columns=['content', 'ozet', 'language', 'dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor', 'content_norm', 'added_by', 'added_date'])
            
            # Create a new row as a dictionary
            new_row = {
                'content': str(news_text),
                'ozet': str(news_text),
                'language': detected_language,  # Use detected language
                'dolar_skor': int(scores['dolar']),
                'altin_skor': int(scores['altin']),
                'borsa_skor': int(scores['borsa']),
                'bitcoin_skor': int(scores['bitcoin']),
                'content_norm': str(news_text),
                'added_by': str(user_name),
                'added_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Convert to DataFrame and append
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)
            
            # Save to Excel
            df.to_excel(excel_file, index=False)
            
            logging.info(f"✅ Added to Excel: {news_text[:50]}... by {user_name}")
            logging.info(f"✅ Fields filled: content, ozet, content_norm, text")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error adding to Excel: {e}")
            return False
    
    def start_polling(self):
        """Start polling for messages"""
        logging.info("🚀 Starting interactive bot...")
        logging.info("📝 Send any text message to get sentiment analysis!")
        logging.info("❓ Send /yardim or /help for usage instructions")
        
        while True:
            try:
                updates = self.get_updates()
                
                for update in updates:
                    update_id = update.get('update_id')
                    message = update.get('message')
                    
                    if update_id > self.last_update_id:
                        self.last_update_id = update_id
                        
                        if message:
                            self.process_message(message)
                
                time.sleep(1)  # Small delay between polls
                
            except KeyboardInterrupt:
                logging.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Polling error: {e}")
                time.sleep(5)

def main():
    """Main function"""
    try:
        # Load config
        config_path = 'bot_config.json'
        if not os.path.exists(config_path):
            logging.error(f"❌ Config file not found: {config_path}")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        bot_token = config.get('bot_token')
        if not bot_token:
            logging.error("❌ Bot token not found in config file")
            return
        
        logging.info(f"✅ Bot token: {bot_token[:20]}...")
        
        # Create bot instance
        bot = InteractiveNewsBot(bot_token)
        
        # Start polling
        bot.start_polling()
        
    except Exception as e:
        logging.error(f"❌ Bot failed to start: {e}")

if __name__ == "__main__":
    main()
