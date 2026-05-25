import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.config import TARGET_COL, ID_COL


def build_features(df: pd.DataFrame):
    df = df.copy()

    drop_cols = [ID_COL]

    # Encode categoricals
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Feature Engineering
    df["CGPA_x_Attendance"] = df["CGPA"] * df["Attendance_Pct"] / 100
    df["Backlog_x_FeeDefault"] = df["Total_Backlogs"] * (df["Fee_Defaults"] + 1)
    df["AttendanceLow"] = (df["Attendance_Pct"] < 60).astype(int)

    feature_cols = [c for c in df.columns if c not in [TARGET_COL, ID_COL]]

    X = df[feature_cols]
    y = df[TARGET_COL]

    return X, y, feature_cols