@echo off
REM ===================================
REM Setup Script for Gemini RAG System (Windows)
REM ===================================

echo.
echo 🚀 Starting Gemini RAG Assistant Setup...
echo.

REM Check Python
echo 📌 Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.9+ from python.org
    pause
    exit /b 1
)
echo ✅ Python found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ℹ️  Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✅ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ Pip upgraded
echo.

REM Install dependencies
echo 📥 Installing dependencies...
echo    This may take a few minutes...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)
echo ✅ Dependencies installed
echo.

REM Create .streamlit directory
echo 📁 Setting up Streamlit configuration...
if not exist ".streamlit" (
    mkdir .streamlit
    echo ✅ .streamlit directory created
) else (
    echo ℹ️  .streamlit directory already exists
)

REM Copy secrets template
if not exist ".streamlit\secrets.toml" (
    if exist ".streamlit\secrets.toml.example" (
        copy .streamlit\secrets.toml.example .streamlit\secrets.toml
        echo ✅ secrets.toml template created
        echo.
        echo ⚠️  IMPORTANT: Edit .streamlit\secrets.toml and add your Gemini API Key!
    )
) else (
    echo ℹ️  secrets.toml already exists
)
echo.

REM Create logs directory
if not exist "logs" (
    mkdir logs
    echo ✅ logs directory created
)
echo.

REM Summary
echo ================================
echo ✨ Setup Complete!
echo ================================
echo.
echo 📋 Next Steps:
echo    1. Edit .streamlit\secrets.toml and add your Gemini API Key
echo    2. Get your API key from: https://ai.google.dev/
echo    3. Run: streamlit run app.py
echo.
echo 📚 Documentation: See README.md for detailed instructions
echo.
echo 🎉 Happy coding!
echo.

pause
