#!/usr/bin/env python3
"""
Cron Job Script for News Fetching and Telegram Notifications
"""

import os
import sys
import json
import time
from datetime import datetime
import logging
from telegram_bot import FinancialNewsBot

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
        logging.FileHandler('logs/cron_news.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config file"""
    try:
        with open('bot_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error("❌ bot_config.json not found")
        return None
    except json.JSONDecodeError:
        logging.error("❌ Invalid JSON in bot_config.json")
        return None

def main():
    """Main function for cron job"""
    logging.info("🚀 Starting cron news fetcher...")
    
    # Load configuration
    config = load_config()
    if not config:
        logging.error("❌ Failed to load configuration")
        sys.exit(1)
    
    # Get bot token and chat ID
    bot_token = config.get('bot_token') or os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = config.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token:
        logging.error("❌ Bot token not found")
        sys.exit(1)
    
    if not chat_id:
        logging.error("❌ Chat ID not found")
        sys.exit(1)
    
    try:
        # Create bot instance
        bot = FinancialNewsBot(bot_token)
        
        # Process news once
        bot.process_news_batch(chat_id)
        
        logging.info("✅ Cron job completed successfully")
        
    except Exception as e:
        logging.error(f"❌ Cron job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
