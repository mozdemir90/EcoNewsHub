#!/usr/bin/env python3
"""
Telegram Bot for Financial News Sentiment Analysis
"""

import os
import requests
import json
import time
import schedule
from datetime import datetime
import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging for telegram bot
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/telegram_bot.log', mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

class FinancialNewsBot:
    def __init__(self, bot_token):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.processed_news = set()  # Keep track of processed news
        self.last_check_time = None
        self.models = {}
        self.vectorizer = None
        self.model_last_modified = {}
        
        # Load processed news from file to persist across restarts
        self.load_processed_news()
        
        # Load models for sentiment analysis
        self.load_models()
    
    def load_processed_news(self):
        """Load processed news from file to avoid duplicates across restarts"""
        try:
            processed_file = 'logs/processed_news.txt'
            if os.path.exists(processed_file):
                with open(processed_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        news_id = line.strip()
                        if news_id:
                            self.processed_news.add(news_id)
                logging.info(f"✅ Loaded {len(self.processed_news)} previously processed news")
            else:
                logging.info("✅ No previous processed news found")
        except Exception as e:
            logging.error(f"❌ Error loading processed news: {e}")
    
    def save_processed_news(self):
        """Save processed news to file to persist across restarts"""
        try:
            processed_file = 'logs/processed_news.txt'
            os.makedirs('logs', exist_ok=True)
            with open(processed_file, 'w', encoding='utf-8') as f:
                for news_id in self.processed_news:
                    f.write(f"{news_id}\n")
            logging.info(f"✅ Saved {len(self.processed_news)} processed news to file")
        except Exception as e:
            logging.error(f"❌ Error saving processed news: {e}")
    
    def get_file_modification_time(self, file_path):
        """Get file modification time"""
        try:
            return os.path.getmtime(file_path)
        except:
            return 0
    
    def should_reload_models(self):
        """Check if models need to be reloaded based on file modification times"""
        model_files = {
            'vectorizer': 'models/tf-idf/tfidf_vectorizer.pkl',
            'dolar_nb': 'models/tf-idf/dolar_skor_nb_model.pkl',
            'altin_nb': 'models/tf-idf/altin_skor_nb_model.pkl', 
            'borsa_nb': 'models/tf-idf/borsa_skor_nb_model.pkl',
            'bitcoin_nb': 'models/tf-idf/bitcoin_skor_nb_model.pkl'
        }
        
        for model_name, file_path in model_files.items():
            current_time = self.get_file_modification_time(file_path)
            last_time = self.model_last_modified.get(model_name, 0)
            
            if current_time > last_time:
                return True
        
        return False
    
    def load_models(self):
        """Load models with modification time tracking"""
        try:
            # Check if models need reloading
            if not self.should_reload_models():
                return
            
            logging.info("🔄 Reloading TF-IDF + NB models...")
            
            # Load TF-IDF vectorizer
            vec_path = 'models/tf-idf/tfidf_vectorizer.pkl'
            if os.path.exists(vec_path):
                self.vectorizer = joblib.load(vec_path)
                self.model_last_modified['vectorizer'] = self.get_file_modification_time(vec_path)
                logging.info("✅ TF-IDF vectorizer loaded")
            else:
                logging.error("❌ TF-IDF vectorizer not found")
                return
            
            # Load Naive Bayes models for each asset
            assets = ['dolar', 'altin', 'borsa', 'bitcoin']
            for asset in assets:
                model_path = f'models/tf-idf/{asset}_skor_nb_model.pkl'
                if os.path.exists(model_path):
                    self.models[asset] = joblib.load(model_path)
                    self.model_last_modified[f'{asset}_nb'] = self.get_file_modification_time(model_path)
                    logging.info(f"✅ {asset} NB model loaded")
                else:
                    logging.warning(f"⚠️ {asset} NB model not found")
            
            logging.info("✅ All TF-IDF + NB models loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Error loading models: {e}")
    
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
        """Predict sentiment for given text using TF-IDF + Naive Bayes models"""
        try:
            # Check if models need reloading before prediction
            if self.should_reload_models():
                self.load_models()
 
            logging.info(f"🔍 Predicting sentiment for text: {text[:100]}...")
            logging.info(f"🔍 Vectorizer loaded: {self.vectorizer is not None}")
            logging.info(f"🔍 Models loaded: {list(self.models.keys())}")
 
            if self.vectorizer is None or not self.models:
                logging.error("❌ Vectorizer or NB models not loaded")
                return None
 
            # Preprocess text
            processed_text = self.preprocess_text(text)
            if not processed_text:
                logging.error("❌ No processed text after preprocessing")
                return None
 
            logging.info(f"🔍 Processed text: {processed_text[:100]}...")
 
            # Vectorize text
            X = self.vectorizer.transform([processed_text])
            X_input = X.toarray()  # NB dense input
 
            # Get predictions for each asset using NB models
            predictions = {}
            asset_names = {
                'dolar': 'USD',
                'altin': 'Gold', 
                'borsa': 'Stock Market',
                'bitcoin': 'Bitcoin'
            }
 
            for asset in ['dolar', 'altin', 'borsa', 'bitcoin']:
                if asset in self.models:
                    pred = self.models[asset].predict(X_input)[0]
                    pred = min(5, max(1, round(pred)))  # Ensure 1-5 range
                    predictions[asset_names[asset]] = pred
                    logging.info(f"✅ {asset}_skor: {pred}/5 (using TF-IDF + NB)")
                else:
                    logging.warning(f"⚠️ Model not found for {asset}")
 
            # Fetch 2 nearest examples from vector DB (if available)
            nearest_text = ""
            try:
                from vector_store import get_vector_store
                vs = get_vector_store()
                if getattr(vs, 'enabled', False):
                    res = vs.query(processed_text, n_results=2)
                    if res:
                        lines = []
                        for item in res:
                            meta = item.get('metadata', {}) or {}
                            sim = None if item.get('distance') is None else round(1 - float(item.get('distance')), 3)
                            snippet = (item.get('document') or '')[:200]
                            lines.append(f"• ID: {item.get('id')} | sim: {sim}\n  D:{meta.get('dolar_skor')} A:{meta.get('altin_skor')} B:{meta.get('borsa_skor')} BTC:{meta.get('bitcoin_skor')}\n  {snippet}")
                        nearest_text = "\n\nBenzer Örnekler:\n" + "\n".join(lines)
            except Exception as e:
                logging.warning(f"Vector fetch failed: {e}")

            logging.info(f"🔍 Final predictions: {predictions}")
            return {"predictions": predictions, "nearest": nearest_text}
            
        except Exception as e:
            logging.error(f"❌ Prediction error: {e}")
            import traceback
            logging.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def get_glove_embeddings(self, text):
        """Get Glove embeddings for text"""
        try:
            import numpy as np
            
            # Split text into words
            words = text.split()
            
            # Get embeddings for each word
            embeddings = []
            for word in words:
                try:
                    if word in self.word2vec_model.wv:
                        embeddings.append(self.word2vec_model.wv[word])
                except:
                    continue
            
            if not embeddings:
                # If no words found, use zero vector
                embedding_size = self.word2vec_model.vector_size
                return np.zeros(embedding_size)
            
            # Average all word embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            return avg_embedding
            
        except Exception as e:
            logging.error(f"❌ Error getting Glove embeddings: {e}")
            return None
    
    def fetch_latest_news(self):
        """Fetch latest news from BloombergHT son dakika only"""
        try:
            logging.info("📰 Fetching BloombergHT son dakika news...")
            
            url = "https://www.bloomberght.com/sondakika"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            news_list = []
            
            # BloombergHT son dakika sayfası yapısına göre haber öğelerini bul
            news_items = soup.find_all("figure", class_="relative")
            logging.info(f"Found {len(news_items)} news items")
            
            if not news_items:
                # Alternatif olarak border-b-2 class'ına sahip div'leri dene
                news_items = soup.find_all("div", class_="border-b-2 border-gray-500 mb-4 pb-4")
                logging.info(f"Found {len(news_items)} news items with border-b-2 selector")
            
            # Get first 5 news
            for i, item in enumerate(news_items[:5]):
                try:
                    # BloombergHT yapısına göre başlığı çıkar
                    title = ""
                    # figcaption içindeki font-unna font-bold text-xl leading-6 class'ına sahip div'i bul
                    title_tag = item.find("div", class_="font-unna font-bold text-xl leading-6")
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                    
                    if not title:
                        # Alternatif olarak figcaption içindeki herhangi bir div'i dene
                        figcaption = item.find("figcaption")
                        if figcaption:
                            title_divs = figcaption.find_all("div")
                            for div in title_divs:
                                text = div.get_text(strip=True)
                                if text and len(text) > 20:  # Anlamlı başlık uzunluğu
                                    title = text
                                    break
                    
                    # BloombergHT yapısına göre link çıkar
                    link = ""
                    # Önce doğrudan haber linkini bulmaya çalış
                    share_links = item.find_all("a", href=True)
                    for share_link in share_links:
                        href = share_link.get("href", "")
                        # Facebook, Twitter gibi sosyal medya linklerini filtrele
                        if any(social in href.lower() for social in ['facebook', 'twitter', 'linkedin', 'whatsapp', 'mailto', 'bluesky', 'bsky', 'wa.me']):
                            continue
                        # Sadece BloombergHT ana linklerini al
                        if "bb.ht" in href or "bloomberght.com" in href:
                            # Sosyal medya paylaşım linklerini filtrele
                            if not any(param in href for param in ['sharer.php', 'intent/tweet', 'sharing/share-offsite', 'intent/compose']):
                                link = href
                                break
                    
                    # Eğer link bulunamazsa, varsayılan son dakika linki kullan
                    if not link:
                        link = "https://www.bloomberght.com/sondakika"
                    
                    # BloombergHT yapısına göre tarih ve saat çıkar
                    date = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    # Saat bilgisini çıkar (text-[#BD1B2E] font-bold text-2xl class'ına sahip span)
                    time_tag = item.find("span", class_="text-[#BD1B2E] font-bold text-2xl")
                    if time_tag:
                        time_text = time_tag.get_text(strip=True)
                        # Tarih bilgisini çıkar (text-xs text-gray-400 class'ına sahip div)
                        date_tag = item.find("div", class_="text-xs text-gray-400")
                        if date_tag:
                            date_text = date_tag.get_text(strip=True)
                            date = f"{date_text} {time_text}"
                        else:
                            date = f"{datetime.now().strftime('%Y-%m-%d')} {time_text}"
                    
                    # BloombergHT için içerik - başlığı kullan, ama daha detaylı yapalım
                    content = title
                    
                    # Eğer mümkünse haber detayını da çek
                    try:
                        if link and "bloomberght.com" in link and link != "https://www.bloomberght.com/sondakika":
                            detail_response = requests.get(link, headers=headers, timeout=10)
                            if detail_response.status_code == 200:
                                detail_soup = BeautifulSoup(detail_response.text, "html.parser")
                                # Haber içeriğini bul
                                content_div = detail_soup.find("div", class_="content")
                                if content_div:
                                    content_text = content_div.get_text(strip=True)
                                    if content_text and len(content_text) > len(title):
                                        content = f"{title}\n\n{content_text[:500]}"  # İlk 500 karakter
                    except:
                        pass  # İçerik çekilemezse başlığı kullan
                    
                    news_data = {
                        "title": title,
                        "url": link,
                        "date": date,
                        "content": content,
                        "source": "BloombergHT Son Dakika"
                    }
                    
                    news_list.append(news_data)
                    logging.info(f"  {i+1}. {title[:60]}...")
                    
                except Exception as e:
                    logging.error(f"  Error processing news {i+1}: {e}")
                    continue
            
            logging.info(f"✅ Successfully fetched {len(news_list)} news articles")
            return news_list
            
        except Exception as e:
            logging.error(f"❌ Error fetching news: {e}")
            return []
    
    def normalize_title(self, title):
        """Normalize title for better comparison"""
        if not title:
            return ""
        
        # Convert to lowercase
        title = title.lower()
        
        # Remove common punctuation and extra spaces
        import re
        title = re.sub(r'[^\w\s]', ' ', title)
        title = re.sub(r'\s+', ' ', title).strip()
        
        # Remove common words that don't add meaning
        stop_words = ['haberi', 'haber', 'son', 'dakika', 'güncel', 'son dakika', 'güncel haber']
        words = title.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words)
    
    def calculate_similarity(self, title1, title2):
        """Calculate similarity between two titles"""
        from difflib import SequenceMatcher
        
        norm1 = self.normalize_title(title1)
        norm2 = self.normalize_title(title2)
        
        # Use sequence matcher for similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity
    
    def filter_new_news(self, news_list):
        """Filter only new news that haven't been processed with improved deduplication"""
        new_news = []
        
        for news in news_list:
            title = news.get('title', '')
            source = news.get('source', '')
            date = news.get('date', '')
            
            # Create multiple identifiers for better deduplication
            news_id_exact = f"{title}_{source}_{date}"
            news_id_normalized = f"{self.normalize_title(title)}_{source}"
            
            # Check exact match first
            if news_id_exact in self.processed_news:
                continue
            
            # Check for similar titles (fuzzy matching)
            is_duplicate = False
            for processed_id in self.processed_news:
                if '_' in processed_id:
                    parts = processed_id.split('_', 2)
                    if len(parts) >= 2:
                        processed_title = parts[0]
                        processed_source = parts[1]
                        
                        # Only compare if same source
                        if processed_source == source:
                            similarity = self.calculate_similarity(title, processed_title)
                            if similarity > 0.8:  # 80% similarity threshold
                                is_duplicate = True
                                logging.info(f"🔄 Duplicate detected: '{title}' ~ '{processed_title}' (similarity: {similarity:.2f})")
                                break
            
            if not is_duplicate:
                new_news.append(news)
                # Store both exact and normalized IDs
                self.processed_news.add(news_id_exact)
                self.processed_news.add(news_id_normalized)
        
        if new_news:
            logging.info(f"📊 Found {len(new_news)} new news articles")
        else:
            logging.info("📊 No new news found")
        
        return new_news
    
    def create_message(self, news, predictions):
        """Create formatted message for Telegram"""
        try:
            # Emoji mapping for sentiment scores
            sentiment_emojis = {
                1: "🔴",  # Very negative
                2: "🟠",  # Negative
                3: "🟡",  # Neutral
                4: "🟢",  # Positive
                5: "🟢"   # Very positive
            }
 
            # Format predictions
            pred_text = ""
            preds = predictions["predictions"] if isinstance(predictions, dict) else predictions
            for asset, score in preds.items():
                emoji = sentiment_emojis.get(score, "⚪")
                pred_text += f"{emoji} {asset}: {score}/5\n"
 
            # Create message
            nearest_block = predictions.get("nearest", "") if isinstance(predictions, dict) else ""
            message = f"""
 📰 **{news.get('source', 'Unknown')}**
 📅 {news.get('date', 'No date')}
 
 **{news.get('title', 'No title')}**
 
 📊 **Analysis (TF-IDF + Naive Bayes):**
 {pred_text}
 
 🤖 **Model:** TF-IDF + Naive Bayes
 🔗 [Read More]({news.get('url', '#')})
{nearest_block}
            """.strip()
            
            return message
            
        except Exception as e:
            logging.error(f"❌ Error creating message: {e}")
            return None
    
    def send_message(self, chat_id, message, disable_web_page_preview=True):
        """Send message to Telegram chat"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': disable_web_page_preview
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
    
    def process_news_batch(self, chat_id):
        """Process news batch and send to Telegram"""
        try:
            logging.info("🔄 Starting news processing...")
            
            # Fetch latest news
            all_news = self.fetch_latest_news()
            if not all_news:
                logging.warning("⚠️ No news fetched")
                return
            
            # Filter new news
            new_news = self.filter_new_news(all_news)
            if not new_news:
                logging.info("📊 No new news found - no message sent")
                return
            
            # Process each news
            for i, news in enumerate(new_news):
                try:
                    logging.info(f"📝 Processing news {i+1}/{len(new_news)}: {news.get('title', '')[:50]}...")
                    
                    predictions = self.predict_sentiment(news.get('content', ''))
                    
                    if predictions:
                        logging.info(f"✅ Predictions received: {predictions}")
                        # Create message
                        message = self.create_message(news, predictions)
                        
                        if message:
                            logging.info(f"✅ Message created, sending to chat {chat_id}")
                            # Send to Telegram
                            success = self.send_message(chat_id, message)
                            if success:
                                logging.info(f"✅ Message sent successfully for news {i+1}")
                            else:
                                logging.error(f"❌ Failed to send message for news {i+1}")
                            
                            # Small delay between messages
                            time.sleep(2)
                        else:
                            logging.error(f"❌ Failed to create message for news {i+1}")
                    else:
                        logging.error(f"❌ No predictions received for news {i+1}")
                    
                except Exception as e:
                    logging.error(f"❌ Error processing news {i+1}: {e}")
                    continue
            
            logging.info(f"✅ Processed {len(new_news)} news articles")
            
            # Save processed news to file
            self.save_processed_news()
            
        except Exception as e:
            logging.error(f"❌ Error in news batch processing: {e}")
    
    def start_scheduler(self, chat_id, interval_minutes=30):
        """Start scheduled news processing"""
        logging.info(f"🚀 Starting scheduler with {interval_minutes} minute intervals")
        
        # Schedule the job
        schedule.every(interval_minutes).minutes.do(self.process_news_batch, chat_id)
        
        # Run immediately once
        self.process_news_batch(chat_id)
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logging.info("🛑 Scheduler stopped by user")
                break
            except Exception as e:
                logging.error(f"❌ Scheduler error: {e}")
                time.sleep(60)

def main():
    """Main function"""
    try:
        # Load config from bot_config.json
        config_path = 'bot_config.json'
        if not os.path.exists(config_path):
            logging.error(f"❌ Config file not found: {config_path}")
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        bot_token = config.get('bot_token')
        chat_id = config.get('chat_id')
        
        if not bot_token:
            logging.error("❌ Bot token not found in config file")
            return
        
        if not chat_id:
            logging.error("❌ Chat ID not found in config file")
            return
        
        logging.info(f"✅ Bot token: {bot_token[:20]}...")
        logging.info(f"✅ Chat ID: {chat_id}")
        
        # Create bot instance
        bot = FinancialNewsBot(bot_token)
        
        # Start scheduler
        interval = config.get('check_interval_minutes', 30)
        logging.info(f"✅ Starting scheduler with {interval} minute intervals")
        bot.start_scheduler(chat_id, interval_minutes=interval)
        
    except Exception as e:
        logging.error(f"❌ Bot failed to start: {e}")

if __name__ == "__main__":
    main()
