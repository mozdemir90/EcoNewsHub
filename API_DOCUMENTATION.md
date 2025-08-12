# Financial News Sentiment Analysis API Documentation

## 🚀 Overview

This API provides sentiment analysis and prediction services for financial news, analyzing the potential impact on USD, Gold, Stock Market, and Bitcoin using multiple machine learning models.

**Base URL:** `http://localhost:5000`

## 📋 Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running and models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "message": "Financial News Sentiment Analysis API is running",
  "models_loaded": true
}
```

---

### 2. Single Prediction

**POST** `/predict`

Analyze a single news text and predict its impact on financial assets.

**Request Body:**
```json
{
  "text": "The Federal Reserve announced a 0.25% interest rate hike, which is expected to strengthen the US dollar against major currencies.",
  "method": "glove",
  "asset": "all"
}
```

**Parameters:**
- `text` (required): The news text to analyze
- `method` (optional): Prediction method - `tfidf`, `word2vec`, or `glove` (default: `glove`)
- `asset` (optional): Specific asset to analyze - `dolar_skor`, `altin_skor`, `borsa_skor`, `bitcoin_skor`, or `all` (default: `all`)

**Response:**
```json
{
  "success": true,
  "text": "The Federal Reserve announced a 0.25% interest rate hike...",
  "method": "glove",
  "predictions": {
    "USD": {
      "rf": 4,
      "svm": 4,
      "ada": 4,
      "ann": 4
    },
    "Gold": {
      "rf": 3,
      "svm": 3,
      "ada": 3,
      "ann": 3
    },
    "Stock Market": {
      "rf": 3,
      "svm": 3,
      "ada": 3,
      "ann": 3
    },
    "Bitcoin": {
      "rf": 3,
      "svm": 3,
      "ada": 3,
      "ann": 3
    }
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Score Interpretation:**
- `1`: Strong negative impact
- `2`: Negative impact
- `3`: Neutral/no impact
- `4`: Positive impact
- `5`: Strong positive impact

---

### 3. Batch Prediction

**POST** `/predict/batch`

Analyze multiple news texts in a single request.

**Request Body:**
```json
{
  "texts": [
    "The Federal Reserve announced a 0.25% interest rate hike.",
    "Bitcoin reached a new all-time high of $50,000.",
    "Gold prices fell due to strong economic data."
  ],
  "method": "glove"
}
```

**Parameters:**
- `texts` (required): Array of news texts to analyze
- `method` (optional): Prediction method (default: `glove`)

**Response:**
```json
{
  "success": true,
  "method": "glove",
  "total_texts": 3,
  "results": [
    {
      "index": 0,
      "text": "The Federal Reserve announced a 0.25% interest rate hike.",
      "predictions": {
        "USD": {"rf": 4, "svm": 4, "ada": 4, "ann": 4},
        "Gold": {"rf": 3, "svm": 3, "ada": 3, "ann": 3},
        "Stock Market": {"rf": 3, "svm": 3, "ada": 3, "ann": 3},
        "Bitcoin": {"rf": 3, "svm": 3, "ada": 3, "ann": 3}
      }
    },
    {
      "index": 1,
      "text": "Bitcoin reached a new all-time high of $50,000.",
      "predictions": {
        "USD": {"rf": 3, "svm": 3, "ada": 3, "ann": 3},
        "Gold": {"rf": 3, "svm": 3, "ada": 3, "ann": 3},
        "Stock Market": {"rf": 3, "svm": 3, "ada": 3, "ann": 3},
        "Bitcoin": {"rf": 5, "svm": 5, "ada": 5, "ann": 5}
      }
    },
    {
      "index": 2,
      "text": "Gold prices fell due to strong economic data.",
      "predictions": {
        "USD": {"rf": 4, "svm": 4, "ada": 4, "ann": 4},
        "Gold": {"rf": 2, "svm": 2, "ada": 2, "ann": 2},
        "Stock Market": {"rf": 4, "svm": 4, "ada": 4, "ann": 4},
        "Bitcoin": {"rf": 3, "svm": 3, "ada": 3, "ann": 3}
      }
    }
  ],
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

### 4. Model Status

**GET** `/models/status`

Get the status of all loaded models.

**Response:**
```json
{
  "success": true,
  "models_status": {
    "tfidf_vectorizer": true,
    "word2vec_model": true,
    "glove_vectors": true,
    "supervised_models": {
      "dolar_skor": {
        "rf": true,
        "svm": true,
        "nb": true,
        "ada": true,
        "ann": true
      },
      "altin_skor": {
        "rf": true,
        "svm": true,
        "nb": true,
        "ada": true,
        "ann": true
      },
      "borsa_skor": {
        "rf": true,
        "svm": true,
        "nb": true,
        "ada": true,
        "ann": true
      },
      "bitcoin_skor": {
        "rf": true,
        "svm": true,
        "nb": true,
        "ada": true,
        "ann": true
      }
    }
  },
  "total_models_loaded": 20
}
```

---

### 5. Reload Models

**POST** `/models/reload`

Reload all models (useful after model updates).

**Response:**
```json
{
  "success": true,
  "message": "Models reloaded successfully",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

---

## 🔧 Usage Examples

### Python Example

```python
import requests
import json

# API base URL
base_url = "http://localhost:5000"

# Single prediction
def predict_sentiment(text, method="glove"):
    url = f"{base_url}/predict"
    data = {
        "text": text,
        "method": method
    }
    
    response = requests.post(url, json=data)
    return response.json()

# Example usage
news_text = "The Federal Reserve announced a 0.25% interest rate hike."
result = predict_sentiment(news_text, "glove")
print(json.dumps(result, indent=2))

# Batch prediction
def predict_batch(texts, method="glove"):
    url = f"{base_url}/predict/batch"
    data = {
        "texts": texts,
        "method": method
    }
    
    response = requests.post(url, json=data)
    return response.json()

# Example batch usage
texts = [
    "Bitcoin reached a new all-time high.",
    "Gold prices fell due to strong economic data.",
    "Stock market rallied on positive earnings."
]

batch_result = predict_batch(texts, "glove")
print(json.dumps(batch_result, indent=2))
```

### JavaScript Example

```javascript
// Single prediction
async function predictSentiment(text, method = 'glove') {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            text: text,
            method: method
        })
    });
    
    return await response.json();
}

// Example usage
const newsText = "The Federal Reserve announced a 0.25% interest rate hike.";
predictSentiment(newsText, 'glove')
    .then(result => console.log(result))
    .catch(error => console.error('Error:', error));

// Batch prediction
async function predictBatch(texts, method = 'glove') {
    const response = await fetch('http://localhost:5000/predict/batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            texts: texts,
            method: method
        })
    });
    
    return await response.json();
}

// Example batch usage
const texts = [
    "Bitcoin reached a new all-time high.",
    "Gold prices fell due to strong economic data.",
    "Stock market rallied on positive earnings."
];

predictBatch(texts, 'glove')
    .then(result => console.log(result))
    .catch(error => console.error('Error:', error));
```

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:5000/health

# Single prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Federal Reserve announced a 0.25% interest rate hike.",
    "method": "glove"
  }'

# Batch prediction
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Bitcoin reached a new all-time high.",
      "Gold prices fell due to strong economic data."
    ],
    "method": "glove"
  }'

# Model status
curl -X GET http://localhost:5000/models/status

# Reload models
curl -X POST http://localhost:5000/models/reload
```

---

## 🚀 Deployment

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
python trainTF-IDF.py
python trainWord2Vec_GloVe.py
python trainDeepLearning.py
```

3. Start the API:
```bash
python api.py
```

### Production Deployment

#### Heroku

1. Create `Procfile`:
```
web: gunicorn api:app
```

2. Create `runtime.txt`:
```
python-3.9.18
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

#### Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "api.py"]
```

2. Build and run:
```bash
docker build -t financial-news-api .
docker run -p 5000:5000 financial-news-api
```

---

## 📊 Model Performance

| Model | Average R² Score | Best Asset | Performance |
|-------|------------------|------------|-------------|
| **GloVe** | -0.1915 | All Assets | ⭐ Best Overall |
| **TF-IDF** | -0.2416 | Bitcoin | ⭐ Best Bitcoin |
| **Word2Vec** | -1.1171 | Gold/Stock | Good for specific assets |

---

## 🔍 Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `500`: Internal Server Error (model loading issues, prediction errors)

Error responses include descriptive messages:

```json
{
  "error": "Text field is required"
}
```

---

## 📞 Support

For issues and questions:
- GitHub Issues: [Project Repository](https://github.com/yourusername/newsFetch)
- Email: your.email@example.com

---

## 📄 License

This API is part of the Financial News Sentiment Analysis project, licensed under MIT License.
