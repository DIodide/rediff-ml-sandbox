#!/bin/bash

# ML Sandbox Environment Setup Script

echo "🚀 Setting up ML Sandbox Environment..."

# Check if Homebrew Python 3.12 is available
PYTHON_CMD="/opt/homebrew/bin/python3.12"
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "❌ Homebrew Python 3.12 is not installed. Please run 'brew install python@3.12' first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment with Python 3.12..."
$PYTHON_CMD -m venv ml_env

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source ml_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create data directories if they don't exist
echo "📁 Creating data directories..."
mkdir -p data/{raw,processed,external}
mkdir -p experiments
mkdir -p logs

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source ml_env/bin/activate"
echo "2. Start Jupyter Lab: jupyter lab"
echo "3. Open notebooks/01_graph_basics.ipynb to get started"
echo ""
echo "📖 For more information, see README.md" 