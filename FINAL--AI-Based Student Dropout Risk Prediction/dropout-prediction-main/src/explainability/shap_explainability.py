import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ShapExplainer:
    """
    Production-ready SHAP wrapper for XGBoost binary classifier.
    """

    def __init__(
        self,
        model_path: str = "xgb_dropout_model.pkl",
        feature_path: str = "feature_cols.pkl",
        output_dir: str = "shap_outputs",
        background_sample_size: int = 1000,
    ):
        self.model = joblib.load(model_path)
        self.feature_cols = joblib.load(feature_path)
        self.output_dir = output_dir
        self.background_sample_size = background_sample_size

        os.makedirs(self.output_dir, exist_ok=True)

        self.explainer = None

    # ─────────────────────────────────────────────
    # Initialize SHAP Explainer
    # ─────────────────────────────────────────────
    def build_explainer(self, X_background: pd.DataFrame):
        if len(X_background) > self.background_sample_size:
            background = X_background.sample(
                self.background_sample_size, random_state=42
            )
        else:
            background = X_background.copy()

        self.explainer = shap.TreeExplainer(
            self.model,
            data=background,
            feature_perturbation="interventional",
        )

    # ─────────────────────────────────────────────
    # Compute SHAP values
    # ─────────────────────────────────────────────
    def compute_shap_values(self, X: pd.DataFrame):
        if self.explainer is None:
            raise RuntimeError("Call build_explainer() before computing SHAP values.")

        X = X[self.feature_cols]
        shap_values = self.explainer(X)

        # Handle binary classification shape
        if len(shap_values.values.shape) == 3:
            return shap_values.values[:, :, 1]
        return shap_values.values

    # ─────────────────────────────────────────────
    # Global Summary Plot
    # ─────────────────────────────────────────────
    def save_summary_plot(self, shap_values, X: pd.DataFrame, max_display=20):
        path = os.path.join(self.output_dir, "shap_summary.png")

        plt.figure()
        shap.summary_plot(
            shap_values,
            X[self.feature_cols],
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    # ─────────────────────────────────────────────
    # Global Bar Importance
    # ─────────────────────────────────────────────
    def save_bar_plot(self, shap_values, X: pd.DataFrame, max_display=20):
        path = os.path.join(self.output_dir, "shap_bar_importance.png")

        plt.figure()
        shap.summary_plot(
            shap_values,
            X[self.feature_cols],
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    # ─────────────────────────────────────────────
    # Top Feature Dependence Plot
    # ─────────────────────────────────────────────
    def save_dependence_plot(self, shap_values, X: pd.DataFrame):
        mean_importance = np.abs(shap_values).mean(axis=0)
        top_feature_index = np.argmax(mean_importance)
        top_feature_name = self.feature_cols[top_feature_index]

        path = os.path.join(self.output_dir, "shap_dependence_top_feature.png")

        plt.figure()
        shap.dependence_plot(
            top_feature_name,
            shap_values,
            X[self.feature_cols],
            show=False,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

        return path

    # ─────────────────────────────────────────────
    # Save Feature Importance Table
    # ─────────────────────────────────────────────
    def save_importance_csv(self, shap_values):
        mean_importance = np.abs(shap_values).mean(axis=0)

        df = pd.DataFrame({
            "feature": self.feature_cols,
            "mean_abs_shap": mean_importance
        }).sort_values("mean_abs_shap", ascending=False)

        path = os.path.join(self.output_dir, "shap_feature_importance.csv")
        df.to_csv(path, index=False)

        return path

    # ─────────────────────────────────────────────
    # Explain Single Student
    # ─────────────────────────────────────────────
    def explain_single(self, student_row: pd.DataFrame):
        if self.explainer is None:
            raise RuntimeError("Call build_explainer() first.")

        student_row = student_row[self.feature_cols]
        shap_values = self.explainer(student_row)

        values = shap_values.values
        if len(values.shape) == 3:
            values = values[:, :, 1]

        contributions = pd.DataFrame({
            "feature": self.feature_cols,
            "shap_value": values[0]
        }).sort_values("shap_value", ascending=False)

        return contributions

    # ─────────────────────────────────────────────
    # Full Pipeline
    # ─────────────────────────────────────────────
    def run_full_analysis(self, X_background: pd.DataFrame, X_test: pd.DataFrame):
        self.build_explainer(X_background)
        shap_values = self.compute_shap_values(X_test)

        artifacts = {
            "summary_plot": self.save_summary_plot(shap_values, X_test),
            "bar_plot": self.save_bar_plot(shap_values, X_test),
            "dependence_plot": self.save_dependence_plot(shap_values, X_test),
            "importance_csv": self.save_importance_csv(shap_values),
        }

        return artifacts