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

# Create a separate logger for cron job
cron_logger = logging.getLogger('cron_news_fetcher')
cron_logger.setLevel(logging.INFO)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Create file handler for cron job
file_handler = logging.FileHandler('logs/cron_news.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger (only file handler, no console handler)
cron_logger.addHandler(file_handler)

def load_config():
    """Load configuration from config file"""
    try:
        with open('bot_config.json', 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        cron_logger.error("❌ bot_config.json not found")
        return None
    except json.JSONDecodeError:
        cron_logger.error("❌ Invalid JSON in bot_config.json")
        return None

def main():
    """Main function for cron job"""
    cron_logger.info("🚀 Starting cron news fetcher...")
    
    # Load configuration
    config = load_config()
    if not config:
        cron_logger.error("❌ Failed to load configuration")
        sys.exit(1)
    
    # Get bot token and chat ID
    bot_token = config.get('bot_token') or os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = config.get('chat_id') or os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token:
        cron_logger.error("❌ Bot token not found")
        sys.exit(1)
    
    if not chat_id:
        cron_logger.error("❌ Chat ID not found")
        sys.exit(1)
    
    try:
        # Create bot instance
        bot = FinancialNewsBot(bot_token)
        
        # Process news once
        bot.process_news_batch(chat_id)
        
        cron_logger.info("✅ Cron job completed successfully")
        
    except Exception as e:
        cron_logger.error(f"❌ Cron job failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
