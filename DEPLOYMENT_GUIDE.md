# 🚀 Deployment Guide

## 📋 Table of Contents

1. [GitHub Pages Deployment](#github-pages-deployment)
2. [API Service Deployment](#api-service-deployment)
3. [SaaS Platform Setup](#saas-platform-setup)
4. [Production Considerations](#production-considerations)

---

## 🌐 GitHub Pages Deployment

### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **Deploy from a branch**
4. Choose **gh-pages** branch and **/(root)** folder
5. Click **Save**

### Step 2: Configure GitHub Actions

The `.github/workflows/deploy.yml` file is already configured to automatically deploy your static demo to GitHub Pages.

### Step 3: Access Your Demo

Your demo will be available at: `https://yourusername.github.io/newsFetch/`

---

## 🔌 API Service Deployment

### Option 1: Heroku Deployment

#### Step 1: Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku

# Windows
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

#### Step 2: Create Heroku App
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-financial-news-api

# Add Python buildpack
heroku buildpacks:set heroku/python
```

#### Step 3: Create Required Files

**Procfile:**
```
web: gunicorn api:app
```

**runtime.txt:**
```
python-3.9.18
```

**requirements.txt (update):**
```
flask==2.3.3
flask-cors==4.0.0
gunicorn==21.2.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
gensim==4.3.1
joblib==1.3.2
```

#### Step 4: Deploy
```bash
# Add all files
git add .

# Commit changes
git commit -m "Add Heroku deployment files"

# Push to Heroku
git push heroku main

# Open the app
heroku open
```

### Option 2: Railway Deployment

#### Step 1: Connect to Railway
1. Go to [Railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select the repository

#### Step 2: Configure Environment
1. Set **Start Command**: `python api.py`
2. Set **Port**: `5000`

#### Step 3: Deploy
Railway will automatically deploy your app when you push to the main branch.

### Option 3: Docker Deployment

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "api.py"]
```

#### Step 2: Build and Run
```bash
# Build the image
docker build -t financial-news-api .

# Run the container
docker run -p 5000:5000 financial-news-api
```

#### Step 3: Deploy to Cloud
```bash
# Tag for registry
docker tag financial-news-api your-registry/financial-news-api:latest

# Push to registry
docker push your-registry/financial-news-api:latest
```

---

## 💼 SaaS Platform Setup

### Step 1: Backend API with Authentication

Create a new file `saas_api.py`:

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///saas.db'
app.config['JWT_SECRET_KEY'] = 'your-secret-key'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(days=1)

db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    api_key = db.Column(db.String(100), unique=True, nullable=False)
    plan = db.Column(db.String(20), default='starter')
    api_calls = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# API usage tracking
class ApiUsage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    endpoint = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    response_time = db.Column(db.Float)
    status_code = db.Column(db.Integer)

@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    user = User(
        email=data['email'],
        password_hash=generate_password_hash(data['password']),
        api_key=f"sk_live_{secrets.token_hex(20)}"
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(email=data['email']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        access_token = create_access_token(identity=user.id)
        return jsonify({'access_token': access_token}), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/api/user/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    
    return jsonify({
        'email': user.email,
        'api_key': user.api_key,
        'plan': user.plan,
        'api_calls': user.api_calls,
        'created_at': user.created_at.isoformat()
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5001)
```

### Step 2: Frontend Dashboard

The SaaS dashboard is already created in `saas_dashboard/index.html`.

### Step 3: Database Setup

```bash
# Install additional dependencies
pip install flask-sqlalchemy flask-jwt-extended

# Run the SaaS API
python saas_api.py
```

---

## 🏭 Production Considerations

### 1. Environment Variables

Create `.env` file:
```env
FLASK_ENV=production
SECRET_KEY=your-super-secret-key
DATABASE_URL=postgresql://user:pass@localhost/dbname
JWT_SECRET_KEY=your-jwt-secret
```

### 2. Security Measures

#### API Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Your prediction logic
    pass
```

#### CORS Configuration
```python
CORS(app, origins=['https://yourdomain.com'], methods=['GET', 'POST'])
```

### 3. Monitoring and Logging

#### Add Logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/financial_news_api.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Financial News API startup')
```

### 4. Performance Optimization

#### Model Caching
```python
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_cached_prediction(text, method):
    cache_key = f"prediction:{hash(text)}:{method}"
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        return pickle.loads(cached_result)
    
    # Make prediction
    result = make_prediction(text, method)
    
    # Cache for 1 hour
    redis_client.setex(cache_key, 3600, pickle.dumps(result))
    return result
```

### 5. Database Optimization

#### PostgreSQL Setup
```sql
-- Create indexes for better performance
CREATE INDEX idx_user_email ON users(email);
CREATE INDEX idx_api_usage_user_id ON api_usage(user_id);
CREATE INDEX idx_api_usage_timestamp ON api_usage(timestamp);

-- Partition tables for large datasets
CREATE TABLE api_usage_2024 PARTITION OF api_usage
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### 6. Load Balancing

#### Nginx Configuration
```nginx
upstream api_servers {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://api_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📊 Monitoring Setup

### 1. Health Checks

```python
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check database connection
        db.session.execute('SELECT 1')
        
        # Check model loading
        if not models:
            raise Exception("Models not loaded")
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'models_loaded': len(models),
            'database': 'connected'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
```

### 2. Metrics Collection

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    try:
        result = make_prediction(request.json['text'])
        REQUEST_COUNT.labels(endpoint='/predict', method='POST').inc()
        return jsonify(result)
    finally:
        REQUEST_DURATION.observe(time.time() - start_time)
```

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring tools configured
- [ ] Backup strategy implemented

### Post-Deployment
- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Monitoring dashboards active
- [ ] Error tracking configured
- [ ] Performance benchmarks met
- [ ] Security scan completed

---

## 📞 Support

For deployment issues:
- Check logs: `heroku logs --tail` (Heroku)
- Monitor metrics in your chosen platform
- Review error tracking tools
- Contact platform support

---

## 🔗 Useful Links

- [Heroku Documentation](https://devcenter.heroku.com/)
- [Railway Documentation](https://docs.railway.app/)
- [Docker Documentation](https://docs.docker.com/)
- [Flask Production Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
