import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.config import MODEL_PATH, FEATURE_PATH, RANDOM_STATE
from src.explainability.shap_explainability import ShapExplainer


def train_model(X, y, feature_cols):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    spw = (y == 0).sum() / (y == 1).sum()

    model = XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURE_PATH)

    # Run SHAP automatically after training
    shap_explainer = ShapExplainer(
        model_path=MODEL_PATH,
        feature_path=FEATURE_PATH
    )

    shap_explainer.run_full_analysis(
        X_background=X_train,
        X_test=X_test
    )

    print("Model + SHAP saved successfully.")