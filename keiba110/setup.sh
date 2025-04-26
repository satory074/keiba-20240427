set -e

echo "Installing Python 3.11.7..."
pyenv install -s 3.11.7

echo "Creating virtual environment..."
pyenv virtualenv 3.11.7 keiba-env
pyenv local keiba-env

echo "Installing Python dependencies..."
pip install -r requirements.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew >/dev/null 2>&1; then
        echo "Installing Java (Temurin) for PDF extraction..."
        brew install --cask temurin
    else
        echo "Homebrew not found. Please install Homebrew first, then run: brew install --cask temurin"
    fi
else
    echo "Installing OpenJDK for PDF extraction..."
    sudo apt-get update
    sudo apt-get install -y openjdk-17-jdk
fi

mkdir -p data/raw/FAILED_RAW data/processed data/model

echo "Setup complete! Activate the environment with: pyenv activate keiba-env"
