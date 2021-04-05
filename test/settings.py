import os
from pathlib import Path

current_path = Path(__file__).resolve().parent

DATA_PATH = os.path.join(current_path, 'data')
CSV_DATA_PATH = os.path.join(DATA_PATH, 'mini.csv')
JSON_DATA_PATH = os.path.join(DATA_PATH, 'mini.json')
TEST_CSV_DATA_PATH = os.path.join(DATA_PATH, 'tmp', 'mini.csv')
TEST_JSON_DATA_PATH = os.path.join(DATA_PATH, 'tmp', 'mini.json')