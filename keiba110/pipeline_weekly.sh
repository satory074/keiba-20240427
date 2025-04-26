
set -e

echo "Running weekly pipeline..."

mkdir -p data/raw data/processed data/model

echo "Fetching data for upcoming races..."
python src/00_fetch/fetch_entries.py
python src/00_fetch/fetch_baba.py
python src/00_fetch/fetch_weather.py

echo "Building features..."
python src/10_feature/build_features.py

echo "Training model..."
python src/20_model/train_stack.py

echo "Weekly pipeline completed successfully!"
