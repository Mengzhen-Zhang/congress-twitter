from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.joinpath('data')
RAW_DATA_ROOT = DATA_ROOT.joinpath('raw')
PROCESSED_DATA_ROOT = DATA_ROOT.joinpath('processed')
SRC_ROOT = PROJECT_ROOT.joinpath('src')
