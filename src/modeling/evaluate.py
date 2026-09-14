"""
src/modeling/evaluate.py

Modul reusable untuk tahap Evaluation project Home Credit Default Risk.
Berisi fungsi memuat model final, menguji SATU KALI di holdout test set,
membuat visualisasi (ROC/PR/confusion matrix), dan interpretasi SHAP.
notebooks/05_evaluation.ipynb HANYA memanggil fungsi di modul ini.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, ConfusionMatrixDisplay

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config
from modeling.train import evaluate_predictions, get_model_dir

FIGURES_DIR = config.MODEL_SUMMARY_PATH.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Fungsi 1: Memuat Model Final dan Data Holdout Test
def load_final_pipeline(model_name: str):
    """Memuat pipeline (preprocessor+model) yang sudah dilatih dan disimpan di tahap Modeling."""
    path = get_model_dir(model_name) / f"{model_name}_pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline {model_name} tidak ditemukan di {path}.")
    return joblib.load(path)


def load_holdout_test_set(
    df: pd.DataFrame,
    split: dict,
    feature_list: list,
    target_column: str = config.TARGET_COLUMN,
    id_column: str = config.ID_COLUMN,
) -> tuple:
    """
    Mengambil HANYA baris test_ids dari holdout_split.json -- data yang TIDAK PERNAH
    disentuh selama CV maupun training. Dipanggil satu kali di akhir siklus Modeling.
    """
    test_ids = split["test_ids"]
    test_df = df[df[id_column].isin(test_ids)]
    X_test = test_df[feature_list]
    y_test = test_df[target_column].values
    return X_test, y_test

# Fungsi 2: Evaluasi Metrik di Holdout Test
def evaluate_on_holdout(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Prediksi SATU KALI di holdout test set, hitung metrik final menggunakan
    evaluate_predictions yang sama persis dengan yang dipakai di CV (konsisten).
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_proba, threshold=threshold)
    metrics["y_test_true"] = np.asarray(y_test).tolist()
    metrics["y_test_proba"] = np.asarray(y_proba).tolist()
    return metrics

# Fungsi 3: Visualisasi ROC Curve, PR Curve, Confusion Matrix
def plot_roc_pr_curves(y_test, y_proba, model_name: str, roc_auc: float, pr_auc: float) -> Path:
    """Membuat side-by-side ROC curve dan Precision-Recall curve, simpan ke reports/figures/."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", color="darkorange")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve — {model_name}")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"PR-AUC = {pr_auc:.4f}", color="teal")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve — {model_name}")
    axes[1].legend()

    plt.tight_layout()
    save_path = FIGURES_DIR / f"{model_name}_roc_pr_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path


def plot_confusion_matrix_heatmap(y_test, y_proba, model_name: str, threshold: float = 0.5) -> Path:
    """Confusion matrix di threshold tertentu, divisualisasikan sebagai heatmap."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Lunas (0)", "Default (1)"],
        cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {model_name} (threshold={threshold})")

    plt.tight_layout()
    save_path = FIGURES_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.show()
    return save_path

# Fungsi 4: Interpretasi Model dengan SHAP

def compute_shap_values(pipeline, X_test: pd.DataFrame, sample_size: int = 2000):
    """
    Menghitung SHAP values pakai TreeExplainer (native & cepat untuk CatBoost).
    sample_size membatasi jumlah baris supaya komputasi tidak terlalu lama --
    2000 baris acak sudah representatif untuk visualisasi summary plot.
    """
    import shap

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    X_test_transformed = preprocessor.transform(X_test)

    if len(X_test_transformed) > sample_size:
        X_sample = X_test_transformed.sample(n=sample_size, random_state=config.RANDOM_SEED)
    else:
        X_sample = X_test_transformed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, model_name: str) -> Path:
    """Beeswarm plot SHAP: fitur mana paling berpengaruh & arah pengaruhnya (naik/turun risiko)."""
    import shap

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Summary — {model_name}")
    plt.tight_layout()

    save_path = FIGURES_DIR / f"{model_name}_shap_summary.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return save_path

# Fungsi 5: Simpan Laporan Evaluasi Final
def save_evaluation_report(metrics: dict, model_name: str) -> Path:
    """Menyimpan metrik holdout test final ke models/{model_name}/{model_name}_holdout_metrics.json."""
    report = {k: v for k, v in metrics.items() if k not in ("y_test_true", "y_test_proba")}
    path = get_model_dir(model_name) / f"{model_name}_holdout_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return path