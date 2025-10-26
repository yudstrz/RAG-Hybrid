#!/bin/bash

# ===================================
# Setup Script for Gemini RAG System
# ===================================

echo "🚀 Starting Gemini RAG Assistant Setup..."
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✅ Pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
echo "   This may take a few minutes..."
pip install -r requirements.txt --quiet
echo "✅ Dependencies installed"
echo ""

# Create .streamlit directory
echo "📁 Setting up Streamlit configuration..."
if [ ! -d ".streamlit" ]; then
    mkdir .streamlit
    echo "✅ .streamlit directory created"
else
    echo "ℹ️  .streamlit directory already exists"
fi

# Copy secrets template if not exists
if [ ! -f ".streamlit/secrets.toml" ]; then
    if [ -f ".streamlit/secrets.toml.example" ]; then
        cp .streamlit/secrets.toml.example .streamlit/secrets.toml
        echo "✅ secrets.toml template created"
        echo ""
        echo "⚠️  IMPORTANT: Edit .streamlit/secrets.toml and add your Gemini API Key!"
    fi
else
    echo "ℹ️  secrets.toml already exists"
fi
echo ""

# Create logs directory
if [ ! -d "logs" ]; then
    mkdir logs
    echo "✅ logs directory created"
fi

# Summary
echo "================================"
echo "✨ Setup Complete!"
echo "================================"
echo ""
echo "📋 Next Steps:"
echo "   1. Edit .streamlit/secrets.toml and add your Gemini API Key"
echo "   2. Get your API key from: https://ai.google.dev/"
echo "   3. Run: streamlit run app.py"
echo ""
echo "📚 Documentation: See README.md for detailed instructions"
echo ""
echo "🎉 Happy coding!"
echo ""
