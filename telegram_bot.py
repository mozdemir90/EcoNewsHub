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

# Configure logging
import os

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Clear any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

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
        
        # Load models for sentiment analysis
        self.load_models()
        
    def load_models(self):
        """Load ML models for sentiment analysis"""
        try:
            import joblib
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Load TF-IDF vectorizer
            self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
            
            # Load models for each asset
            self.models = {}
            assets = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
            model_types = ['rf', 'svm', 'nb', 'ada', 'ann']
            
            for asset in assets:
                self.models[asset] = {}
                for model_type in model_types:
                    try:
                        model_path = f'models/{asset}_{model_type}_model.pkl'
                        if os.path.exists(model_path):
                            self.models[asset][model_type] = joblib.load(model_path)
                    except Exception as e:
                        logging.warning(f"Could not load {asset}_{model_type}: {e}")
            
            logging.info("✅ Models loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Error loading models: {e}")
            self.vectorizer = None
            self.models = {}
    
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
        """Predict sentiment for given text using best models for each asset"""
        try:
            if not self.vectorizer or not self.models:
                return None
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return None
            
            # Transform text
            X = self.vectorizer.transform([processed_text])
            
            # Get predictions for each asset using best models
            predictions = {}
            asset_names = {
                'dolar_skor': 'USD',
                'altin_skor': 'Gold', 
                'borsa_skor': 'Stock Market',
                'bitcoin_skor': 'Bitcoin'
            }
            
            # Best models for each asset based on performance comparison
            best_models = {
                'dolar_skor': 'rf',      # Word2Vec RF was best, but we'll use TF-IDF RF for consistency
                'altin_skor': 'rf',      # GloVe RF was best, but we'll use TF-IDF RF for consistency
                'borsa_skor': 'rf',      # Word2Vec RF was best, but we'll use TF-IDF RF for consistency
                'bitcoin_skor': 'rf'     # TF-IDF RF was best
            }
            
            for asset, models in self.models.items():
                best_model = best_models.get(asset, 'rf')
                if best_model in models:
                    pred = models[best_model].predict(X)[0]
                    pred = min(5, max(1, round(pred)))  # Ensure 1-5 range
                    predictions[asset_names[asset]] = pred
                    logging.info(f"✅ {asset}: {pred}/5 (using {best_model})")
                else:
                    logging.warning(f"⚠️ {best_model} model not found for {asset}")
            
            return predictions
            
        except Exception as e:
            logging.error(f"❌ Prediction error: {e}")
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
                    
                    # BloombergHT için içerik - başlığı kullan
                    content = title
                    
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
    
    def filter_new_news(self, news_list):
        """Filter only new news that haven't been processed"""
        new_news = []
        
        for news in news_list:
            # Create unique identifier for news
            news_id = f"{news.get('title', '')}_{news.get('source', '')}_{news.get('date', '')}"
            
            if news_id not in self.processed_news:
                new_news.append(news)
                self.processed_news.add(news_id)
        
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
            for asset, score in predictions.items():
                emoji = sentiment_emojis.get(score, "⚪")
                pred_text += f"{emoji} {asset}: {score}/5\n"
            
            # Create message
            message = f"""
📰 **{news.get('source', 'Unknown')}**
📅 {news.get('date', 'No date')}

**{news.get('title', 'No title')}**

📊 **Sentiment Analysis:**
{pred_text}

🔗 [Read More]({news.get('url', '#')})
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
            for news in new_news:
                try:
                    # Get sentiment prediction
                    predictions = self.predict_sentiment(news.get('content', ''))
                    
                    if predictions:
                        # Create message
                        message = self.create_message(news, predictions)
                        
                        if message:
                            # Send to Telegram
                            self.send_message(chat_id, message)
                            
                            # Small delay between messages
                            time.sleep(2)
                    
                except Exception as e:
                    logging.error(f"❌ Error processing news: {e}")
                    continue
            
            logging.info(f"✅ Processed {len(new_news)} news articles")
            
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
