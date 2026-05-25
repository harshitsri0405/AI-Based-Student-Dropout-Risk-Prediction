from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw/student_dropout_v6.xlsx"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_dropout_model.pkl"
FEATURE_PATH = MODEL_DIR / "feature_cols.pkl"
SHAP_OUTPUT_DIR = MODEL_DIR / "shap_outputs"

TARGET_COL = "Dropout"
ID_COL = "Student_ID"
RANDOM_STATE = 42