# Financial News Sentiment Analysis & Prediction System

A comprehensive machine learning system that analyzes financial news and predicts their impact on financial assets (USD, Gold, Stock Market, Bitcoin) using multiple NLP and ML approaches.

## 🚀 Features

### Supervised Learning Models
- **TF-IDF + Regression Models**
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

### Deep Learning Models
- **1D CNN** - Convolutional Neural Network
- **LSTM** - Long Short-Term Memory
- **CNN + LSTM** - Hybrid model

## 📊 Model Performance Comparison

| Model | Average R² Score | Best Asset | Performance |
|-------|------------------|------------|-------------|
| **GloVe** | -0.1915 | All Assets | ⭐ Best Overall |
| **TF-IDF** | -0.2416 | Bitcoin | ⭐ Best Bitcoin |
| **Word2Vec** | -1.1171 | Gold/Stock | Good for specific assets |
| **Deep Learning** | -22.5117 | Needs improvement | 🔧 Under Development |

## 📁 Project Structure

```
newsFetch/
├── app.py                          # Flask web application
├── trainTF-IDF.py                  # TF-IDF models training
├── trainWord2Vec_GloVe.py          # Word2Vec/GloVe models training
├── trainDeepLearning.py            # Deep Learning models training
├── compare_models.py               # Model comparison script
├── labelNews.py                    # News labeling and scoring
├── news_fetcher.py                 # News scraping script
├── prep.py                         # Data preprocessing
├── requirements.txt                # Required libraries
├── data/                          # Data files
│   ├── training_data.xlsx         # Training data
│   ├── analiz_sonuclari2.xlsx    # Test data
│   └── model_comparison.xlsx     # Model comparison results
├── models/                        # Trained models
│   ├── *.pkl                     # Supervised learning models
│   └── deep_learning/            # Deep learning models
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

3. **Prepare data and train models:**
```bash
python labelNews.py
python trainTF-IDF.py
python trainWord2Vec_GloVe.py
python trainDeepLearning.py
```

4. **Start the web application:**
```bash
python app.py
```

The application will be available at `http://localhost:5050`

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

1. Enter news text in the web interface
2. Select method (TF-IDF, Word2Vec, GloVe, Deep Learning)
3. Choose model
4. Click "Predict" button
5. View results

## 📝 Scoring System

- **1:** Strong negative impact
- **2:** Negative impact
- **3:** Neutral/no impact
- **4:** Positive impact
- **5:** Strong positive impact

## 🎯 Key Features

- **Multi-source news scraping** from financial websites
- **Multi-model approach** for robust predictions
- **Real-time web interface** for easy interaction
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

## 🚀 Deployment Options

### GitHub Pages (Static Demo)
For a static demo version, you can deploy the web interface using GitHub Pages.

### Heroku/Railway (Full Application)
Deploy the complete Flask application with model serving capabilities.

### API Service
Convert the prediction functionality into a REST API for integration with other applications.

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **GitHub:** [@mozdemir90](https://github.com/yourusername)
- **LinkedIn:** [muammer-özdemir-05629933](https://linkedin.com/in/yourprofile)
- **Email:** your.email@example.com

## 🙏 Acknowledgments

- Financial news sources for providing data
- Open-source ML/NLP libraries
- Community contributors and feedback

---

⭐ **Star this repository if you find it useful!** 