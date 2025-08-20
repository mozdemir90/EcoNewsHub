# Financial News Sentiment Analysis & Prediction System

A comprehensive machine learning system that analyzes financial news and predicts their impact on financial assets (USD, Gold, Stock Market, Bitcoin) using multiple NLP and ML approaches. Features include automated news fetching, Telegram bot integration, and organized model management.

## 🚀 Features

### 🤖 Interactive Telegram Bot
- **Real-time news analysis** via Telegram
- **Interactive scoring** for training data collection
- **Automated news fetching** every 10 minutes
- **Multi-model predictions** with color-coded results

### 📊 Supervised Learning Models
- **TF-IDF + Regression Models** (Best Performance)
  - Random Forest
  - Support Vector Machine (SVM)
  - Artificial Neural Network (ANN)
  - AdaBoost
  - Naive Bayes

- **Word2Vec + Regression Models**
  - Random Forest
  - SVM
  - ANN
  - AdaBoost

- **GloVe + Regression Models**
  - Random Forest
  - SVM
  - ANN
  - AdaBoost

### 🧠 Deep Learning Models
- **1D CNN** - Convolutional Neural Network
- **LSTM** - Long Short-Term Memory
- **CNN + LSTM** - Hybrid model

## 📊 Model Performance Comparison

| Model | Average R² Score | Best Asset | Performance |
|-------|------------------|------------|-------------|
| **TF-IDF** | -0.1724 | Bitcoin (R² = 0.1745) | ⭐ Best Overall |
| **GloVe** | -0.1782 | Altın/Borsa | Good Performance |
| **Word2Vec** | -0.8371 | Dolar | Moderate Performance |
| **Deep Learning** | -13.1799 | Needs improvement | 🔧 Under Development |

### 🏆 Best Performance by Asset:
- **Bitcoin**: TF-IDF (R² = 0.1745) - **Positive Score!**
- **Dolar**: Word2Vec (R² = -0.3599)
- **Altın**: GloVe (R² = -0.0304)
- **Borsa**: GloVe (R² = -0.0650)

## 📁 Project Structure

```
newsFetch/
├── app.py                          # Flask web application
├── telegram_interactive_bot.py     # Interactive Telegram bot
├── telegram_bot.py                 # Automated Telegram bot
├── cron_news_fetcher.py           # Automated news fetcher
├── trainTF-IDF.py                  # TF-IDF models training
├── trainWord2Vec_GloVe.py          # Word2Vec/GloVe models training
├── trainDeepLearning.py            # Deep Learning models training
├── compare_models.py               # Model comparison script
├── labelNews.py                    # News labeling and scoring
├── news_fetcher.py                 # News scraping script
├── prep.py                         # Data preprocessing
├── requirements.txt                # Required libraries
├── bot_config.json                 # Telegram bot configuration
├── data/                          # Data files
│   ├── training_data.xlsx         # Training data
│   ├── training_data_telegram.json # Telegram training data
│   ├── analiz_sonuclari2.xlsx    # Test data
│   └── model_comparison.xlsx     # Model comparison results
├── models/                        # Organized model directories
│   ├── tf-idf/                   # TF-IDF models
│   ├── word2vec/                 # Word2Vec models
│   ├── glove/                    # GloVe models
│   ├── deeplearning/             # Deep learning models
│   └── README.md                 # Model documentation
├── logs/                         # System logs
│   ├── interactive_bot.log       # Telegram bot logs
│   ├── cron_news.log            # Automated news logs
│   └── user_actions.log         # User activity logs
└── templates/                     # Web interface
    ├── index.html
    └── ekle.html
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/newsFetch.git
cd newsFetch
```

2. **Install required libraries:**
```bash
pip install -r requirements.txt
```

3. **Configure Telegram Bot:**
```bash
# Copy the example config file
cp bot_config.example.json bot_config.json

# Edit bot_config.json with your bot token
{
    "bot_token": "YOUR_BOT_TOKEN_HERE",
    "chat_id": "YOUR_CHAT_ID_HERE",
    "admin_users": ["YOUR_USERNAME_HERE"]
}
```

4. **Prepare data and train models:**
```bash
python labelNews.py
python trainTF-IDF.py
python trainWord2Vec_GloVe.py
python trainDeepLearning.py
```

5. **Start the web application:**
```bash
python app.py
```

The application will be available at `http://localhost:5050`

## 🤖 Telegram Bot Setup

### Interactive Bot
```bash
python telegram_interactive_bot.py
```

**Commands:**
- Send any text message for sentiment analysis
- `/help` or `/yardim` - Show usage instructions
- `/ekle <text> | dolar:3 altin:4 borsa:2 bitcoin:5` - Add training data
- `/ekle_interactive <text>` - Interactive scoring mode

### Automated News Bot
```bash
# Set up cron job for automated news fetching
crontab -e
# Add: */10 * * * * cd /path/to/newsFetch && python3 cron_news_fetcher.py >> logs/cron_news.log 2>&1
```

## 📊 Model Comparison

Compare all models performance:
```bash
python compare_models.py
```

This script generates:
- `data/model_comparison.xlsx` - Comparison table
- `data/model_comparison_heatmap.png` - Performance heatmap
- `data/r2_comparison.png` - R² scores graph
- `data/model_comparison_report.md` - Detailed report

## 🌐 Web Interface

- **Main Page:** Make news predictions
- **Add to Training Set:** Add new news with scores
- **Model Options:**
  - TF-IDF, Word2Vec, GloVe (Supervised Learning)
  - Deep Learning (CNN, LSTM, CNN+LSTM)

## 📈 Performance Metrics

- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (R-squared)
- **RMSE** (Root Mean Squared Error)

## 🔧 Usage

### Web Interface
1. Enter news text in the web interface
2. Select method (TF-IDF, Word2Vec, GloVe, Deep Learning)
3. Choose model
4. Click "Predict" button
5. View results

### Telegram Bot
1. Send news text to the bot
2. Receive instant sentiment analysis
3. Use interactive commands for training data

## 📝 Scoring System

- **1:** Strong negative impact 🔴
- **2:** Negative impact 🔴
- **3:** Neutral/no impact 🟡
- **4:** Positive impact 🟢
- **5:** Strong positive impact 🟢

## 🎯 Key Features

- **Multi-source news scraping** from financial websites
- **Multi-model approach** for robust predictions
- **Real-time web interface** for easy interaction
- **Interactive Telegram bot** for mobile access
- **Automated news fetching** every 10 minutes
- **Organized model management** with clear directory structure
- **Comprehensive model comparison** with detailed metrics
- **Extensible architecture** for adding new models

## 🔬 Technical Details

### Data Sources
- Bloomberg HT
- CNN Business
- BBC News
- Anadolu Agency
- Hürriyet Daily News
- And more...

### Technologies Used
- **Backend:** Python, Flask
- **ML/NLP:** Scikit-learn, TensorFlow, Gensim, NLTK
- **Data Processing:** Pandas, NumPy
- **Web Scraping:** BeautifulSoup, Requests
- **Frontend:** HTML, CSS, JavaScript
- **Telegram API:** python-telegram-bot
- **Automation:** Cron jobs

## 🚀 Deployment Options

### GitHub Pages (Static Demo)
For a static demo version, you can deploy the web interface using GitHub Pages.

### Heroku/Railway (Full Application)
Deploy the complete Flask application with model serving capabilities.

### API Service
Convert the prediction functionality into a REST API for integration with other applications.

### Telegram Bot Hosting
- **Railway:** Easy deployment with automatic scaling
- **Heroku:** Reliable hosting for Telegram bots
- **VPS:** Full control over the environment

## 📋 Recent Updates (August 2025)

### ✅ Model Organization
- **Reorganized models** into dedicated directories (tf-idf/, word2vec/, glove/, deeplearning/)
- **Updated all paths** in training and prediction scripts
- **Fixed model loading** issues across all applications

### ✅ Telegram Bot Improvements
- **Interactive bot** with real-time sentiment analysis
- **Automated news fetcher** with cron job integration
- **Color-coded results** for better user experience
- **Training data collection** via Telegram commands

### ✅ Performance Optimizations
- **Switched to TF-IDF models** for better prediction accuracy
- **Fixed duplicate detection** in news fetching
- **Improved error handling** and logging
- **Enhanced user interface** with better feedback

### ✅ System Monitoring
- **Comprehensive logging** for all components
- **Real-time monitoring** of bot activities
- **Performance tracking** for model comparisons
- **Automated backup** of training data

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **GitHub:** [@mozdemir90](https://github.com/mozdemir90)
- **LinkedIn:** [muammer özdemir](https://linkedin.com/in/muammer-özdemir-05629933)
- **Email:** your.email@example.com

## 🙏 Acknowledgments

- Financial news sources for providing data
- Open-source ML/NLP libraries
- Community contributors and feedback
- Telegram Bot API for interactive features

---

⭐ **Star this repository if you find it useful!** 

🚀 **Ready to analyze financial news with AI-powered predictions!** 