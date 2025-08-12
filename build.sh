#!/bin/bash

# Build script for Render deployment

echo "🚀 Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install system dependencies (if needed)
# apt-get update && apt-get install -y gfortran

# Install Python packages
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# Download models (if needed)
echo "📥 Downloading models..."
python download_models.py

echo "✅ Build completed!"
