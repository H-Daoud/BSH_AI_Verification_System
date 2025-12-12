#!/bin/bash
echo "🚀 Setting up BSH Verification System..."

# 1. Setup Python Environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

echo "⬇️ Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python setup complete!"
echo ""
echo "👉 To run the Backend API:"
echo "   source venv/bin/activate"
echo "   uvicorn 05_backend_system.api_service.app.main:app --reload"
echo ""
echo "👉 To run the Frontend (Flutter):"
echo "   cd 06_frontend_interactive"
echo "   flutter pub get"
echo "   flutter run"
