"""
src/modeling/train.py

Modul reusable untuk tahap Modeling project Home Credit Default Risk.
notebooks/04_modeling.ipynb HANYA memanggil fungsi di modul ini — tidak
boleh menulis ulang logika training/preprocessing langsung di dalam cell.
"""

import sys
import json
import random
import time
from pathlib import Path
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, log_loss, confusion_matrix, accuracy_score,
)

# Bootstrap: pastikan folder src/ ada di sys.path agar `import config` selalu berhasil.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
import config


CATEGORICAL_COLUMNS = [
    "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "CODE_GENDER", "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE", "HOUSETYPE_MODE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR",
    "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "FLAG_OWN_REALTY",
]


def set_random_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set seed random & numpy (dan torch bila terpasang) agar eksperimen reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_featured_dataset(path: Path = config.FEATURED_DATASET_PATH) -> pd.DataFrame:
    """Load dataset hasil Feature Engineering final (307.511 baris, 73 kolom)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset featured tidak ditemukan di {path}. "
            "Pastikan notebooks/03_feature_engineering.ipynb sudah dijalankan."
        )
    return pd.read_csv(path)


def load_feature_metadata(path: Path = config.FEATURE_METADATA_PATH) -> dict:
    """Load feature_engineering_metadata.json: tree_features, linear_mlp_features, decision_log."""
    if not path.exists():
        raise FileNotFoundError(f"File metadata tidak ditemukan di {path}.")
    with open(path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    required_keys = ["tree_features", "linear_mlp_features"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise KeyError(f"Key metadata berikut tidak ditemukan di {path}: {missing}")
    return metadata


def create_or_load_holdout_split(
    df: pd.DataFrame,
    id_column: str = config.ID_COLUMN,
    target_column: str = config.TARGET_COLUMN,
    test_size: float = config.TEST_SIZE,
    seed: int = config.RANDOM_SEED,
    split_path: Path = config.HOLDOUT_SPLIT_PATH,
) -> dict:
    """Buat (sekali) atau muat stratified holdout split 80% dev / 20% test, freeze ke JSON."""
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ids = df[id_column].values
    y = df[target_column].values
    dev_ids, test_ids = train_test_split(ids, test_size=test_size, stratify=y, random_state=seed)

    split = {
        "random_seed": seed,
        "test_size": test_size,
        "development_ids": dev_ids.tolist(),
        "test_ids": test_ids.tolist(),
        "development_size": len(dev_ids),
        "test_size_count": len(test_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    return split


def create_or_load_cv_folds(
    df: pd.DataFrame,
    split: dict,
    id_column: str = config.ID_COLUMN,
    target_column: str = config.TARGET_COLUMN,
    n_splits: int = config.N_SPLITS,
    seed: int = config.RANDOM_SEED,
    folds_path: Path = config.CV_FOLDS_PATH,
) -> dict:
    """Buat (sekali) atau muat Stratified n-Fold pada development set, freeze ke JSON."""
    if folds_path.exists():
        with open(folds_path, "r", encoding="utf-8") as f:
            return json.load(f)

    dev_ids = split["development_ids"]
    dev_df = df.set_index(id_column).loc[dev_ids].reset_index()
    y_dev = dev_df[target_column].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_assignment = np.full(len(dev_ids), fill_value=-1, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(dev_df, y_dev)):
        fold_assignment[val_idx] = fold_idx
    assert (fold_assignment >= 0).all(), "Ada baris development yang tidak masuk fold manapun."

    folds = {
        "random_seed": seed,
        "n_splits": n_splits,
        "development_ids": dev_ids,
        "fold_assignment": fold_assignment.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    folds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=2)
    return folds


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Kategori dengan frekuensi < threshold pada data fit (fold-train) dikelompokkan jadi 'Other'."""

    def __init__(self, threshold: int = config.RARE_CATEGORY_THRESHOLD, other_label: str = "Other"):
        self.threshold = threshold
        self.other_label = other_label

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.frequent_categories_ = {
            col: X[col].value_counts()[lambda s: s >= self.threshold].index.tolist()
            for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            allowed = self.frequent_categories_.get(col, [])
            X[col] = X[col].where(X[col].isin(allowed), self.other_label)
        return X


def _split_categorical_numeric(feature_list: list) -> tuple:
    categorical_cols = [c for c in feature_list if c in CATEGORICAL_COLUMNS]
    numeric_cols = [c for c in feature_list if c not in CATEGORICAL_COLUMNS]
    return categorical_cols, numeric_cols


def build_linear_mlp_preprocessor(
    feature_list: list, rare_threshold: int = config.RARE_CATEGORY_THRESHOLD
) -> ColumnTransformer:
    """Preprocessing Logistic Regression & MLP: RareCategoryGrouper + OneHot untuk kategorikal,
    SimpleImputer(median) + StandardScaler untuk numerik (menangani NaN dari fitur agregat v2)."""
    categorical_cols, numeric_cols = _split_categorical_numeric(feature_list)
    categorical_pipeline = Pipeline([
        ("rare_grouper", RareCategoryGrouper(threshold=rare_threshold)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipeline, categorical_cols),
        ("num", numeric_pipeline, numeric_cols),
    ])

def build_tree_preprocessor(
    feature_list: list, rare_threshold: int = config.RARE_CATEGORY_THRESHOLD
) -> ColumnTransformer:
    """Preprocessing Decision Tree/Random Forest/XGBoost: RareCategoryGrouper + OneHot,
    tanpa scaler, memakai fitur raw (bukan LOG)."""
    categorical_cols, numeric_cols = _split_categorical_numeric(feature_list)
    categorical_pipeline = Pipeline([
        ("rare_grouper", RareCategoryGrouper(threshold=rare_threshold)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("cat", categorical_pipeline, categorical_cols),
        ("num", "passthrough", numeric_cols),
    ])


def compute_scale_pos_weight(y) -> float:
    """
    Menghitung scale_pos_weight untuk LightGBM/XGBoost: rasio jumlah kelas
    negatif terhadap kelas positif. HARUS dihitung dari data training saja
    (fold-train atau development set), bukan fold-validation/holdout test.
    """
    y = np.asarray(y)
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    if n_positive == 0:
        raise ValueError("Tidak ada sampel TARGET=1 di data training ini.")
    return float(n_negative / n_positive)


def evaluate_predictions(y_true, y_proba, threshold: float = 0.5) -> dict:
    """
    Metrik evaluasi standar: ROC-AUC & PR-AUC dari probabilitas (tanpa threshold),
    precision/recall/F1/confusion matrix pada threshold 0.5 (Opsi A — hanya untuk
    perbandingan kandidat di tahap CV, bukan threshold bisnis final).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold": threshold,
    }

import joblib
import csv


def get_model_dir(model_name: str) -> Path:
    """Membuat (jika belum ada) dan mengembalikan folder models/{model_name}/."""
    model_dir = config.MODELS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_model_pipeline(pipeline, model_name: str) -> Path:
    """Menyimpan pipeline/model terlatih sebagai models/{model_name}/{model_name}_pipeline.joblib."""
    model_dir = get_model_dir(model_name)
    path = model_dir / f"{model_name}_pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


def save_model_json(data: dict, model_name: str, suffix: str) -> Path:
    """
    Menyimpan dict sebagai JSON: models/{model_name}/{model_name}_{suffix}.json.
    Dipakai untuk history, metrics, dan config — suffix membedakan jenisnya.
    """
    model_dir = get_model_dir(model_name)
    path = model_dir / f"{model_name}_{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


MODEL_SUMMARY_COLUMNS = [
    "model_name", "run_id", "status", "feature_path",
    "n_features_before_encoding", "n_features_after_encoding",
    "cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean", "cv_pr_auc_std",
    "best_iteration_mean", "training_time_seconds", "random_seed",
    "model_path", "history_path", "timestamp",
]


def update_model_summary(row: dict, summary_path: Path = config.MODEL_SUMMARY_PATH) -> None:
    """
    Menambahkan satu baris eksperimen ke reports/model_summary.csv dengan SKEMA KOLOM TETAP
    (MODEL_SUMMARY_COLUMNS) -- supaya semua model (boosting maupun non-boosting) selalu align
    di kolom yang sama. Key yang tidak ada di dict suatu model (mis. best_iteration_mean untuk
    Logistic Regression) otomatis diisi kosong oleh DictWriter, bukan menggeser kolom lain.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_SUMMARY_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Fungsi Dasar: Fit dan Evaluasi Satu Fold
# Menerima model dan preprocessor yang belum di-fit, fit hanya pada fold-train, 
# transform fold-validation dengan parameter dari fold-train (anti-leakage), 
# lalu evaluasi.

def _fit_and_evaluate_fold(model, preprocessor, X_train, y_train, X_val, y_val, threshold: float = 0.5) -> dict:
    """Fit preprocessor+model pada fold-train, evaluasi pada fold-validation."""
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    model.fit(X_train_transformed, y_train)
    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]

    metrics = evaluate_predictions(y_val, y_val_proba, threshold=threshold)
    metrics["y_val_true"] = np.asarray(y_val).tolist()
    metrics["y_val_proba"] = np.asarray(y_val_proba).tolist()
    return metrics

# Loop CV Lengkap, Generik untuk Model Non-Boosting
# Membuat preprocessor dan model baru (fresh, belum fit) di setiap fold — mencegah 
# kebocoran state dari fold sebelumnya. Mengumpulkan metrik per fold sekaligus 
# probabilitas out-of-fold.

def run_stratified_cv(
    df: pd.DataFrame,
    folds: dict,
    feature_list: list,
    target_column: str,
    preprocessor_builder,
    model_factory,
    id_column: str = config.ID_COLUMN,
    threshold: float = 0.5,
) -> dict:
    """Loop Stratified n-Fold: fit fresh preprocessor+model tiap fold, kumpulkan metrik dan OOF proba."""
    dev_ids = np.array(folds["development_ids"])
    fold_arr = np.array(folds["fold_assignment"])
    n_splits = folds["n_splits"]

    dev_df = df.set_index(id_column).loc[dev_ids].reset_index()

    fold_metrics = []
    oof_true = np.full(len(dev_ids), fill_value=np.nan)
    oof_proba = np.full(len(dev_ids), fill_value=np.nan)

    for fold_idx in range(n_splits):
        train_mask = fold_arr != fold_idx
        val_mask = fold_arr == fold_idx

        X_train = dev_df.loc[train_mask, feature_list]
        y_train = dev_df.loc[train_mask, target_column].values
        X_val = dev_df.loc[val_mask, feature_list]
        y_val = dev_df.loc[val_mask, target_column].values

        preprocessor = preprocessor_builder(feature_list)
        model = model_factory()

        result = _fit_and_evaluate_fold(model, preprocessor, X_train, y_train, X_val, y_val, threshold=threshold)

        fold_result = {k: v for k, v in result.items() if k not in ("y_val_true", "y_val_proba")}
        fold_result["fold"] = fold_idx
        fold_metrics.append(fold_result)

        oof_true[val_mask] = result["y_val_true"]
        oof_proba[val_mask] = result["y_val_proba"]

    return {"fold_metrics": fold_metrics, "oof_true": oof_true, "oof_proba": oof_proba}

# Agregasi Mean/Std dan Retrain Final

def aggregate_cv_metrics(fold_metrics: list) -> dict:
    """Hitung mean & std tiap metrik numerik dari list hasil per fold."""
    exclude_keys = {"fold", "confusion_matrix", "threshold"}
    metric_keys = [k for k in fold_metrics[0].keys() if k not in exclude_keys]
    aggregated = {}
    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated


def retrain_on_full_development(
    df: pd.DataFrame,
    split: dict,
    feature_list: list,
    target_column: str,
    preprocessor_builder,
    model_factory,
    id_column: str = config.ID_COLUMN,
) -> Pipeline:
    """Fit satu Pipeline (preprocessor+model) pada SELURUH development set. Dipanggil setelah CV selesai."""
    dev_ids = split["development_ids"]
    dev_df = df[df[id_column].isin(dev_ids)]

    X_dev = dev_df[feature_list]
    y_dev = dev_df[target_column].values

    pipeline = Pipeline([
        ("preprocessor", preprocessor_builder(feature_list)),
        ("model", model_factory()),
    ])
    pipeline.fit(X_dev, y_dev)
    return pipeline

# Orchestrator: train_logistic_regression
# Ini fungsi tipis — hanya memanggil fungsi-fungsi di atas dalam urutan yang benar, 
# tidak berisi logika CV/fit sendiri.

import time
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(df: pd.DataFrame, metadata: dict, split: dict, folds: dict, seed: int = config.RANDOM_SEED) -> dict:
    """Orchestrator Logistic Regression: CV -> agregasi -> retrain -> simpan artifact -> update summary."""
    model_name = "logistic_regression"
    feature_list = metadata["linear_mlp_features"]
    target_column = config.TARGET_COLUMN

    model_factory = lambda: LogisticRegression(class_weight="balanced", random_state=seed, max_iter=1000)

    start_time = time.time()
    cv_result = run_stratified_cv(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_linear_mlp_preprocessor,
        model_factory=model_factory,
        threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_linear_mlp_preprocessor,
        model_factory=model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(split["development_ids"])][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json({"fold_metrics": cv_result["fold_metrics"]}, model_name, suffix="history")
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {"class_weight": "balanced", "random_state": seed, "max_iter": 1000, "feature_path": "linear_mlp"},
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name,
        "run_id": f"{model_name}_v1",
        "status": "done",
        "feature_path": "linear_mlp",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"],
        "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"],
        "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "training_time_seconds": training_time,
        "random_seed": seed,
        "model_path": str(pipeline_path),
        "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"],
        "cv_summary": cv_summary,
        "pipeline": final_pipeline,
        "oof_true": cv_result["oof_true"],
        "oof_proba": cv_result["oof_proba"],
    }

# Decision Tree
# Karena infrastrukturnya sudah generik, train_decision_tree() jauh lebih ringkas — hanya beda model_factory, feature_list, dan preprocessor_builder.

from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(df: pd.DataFrame, metadata: dict, split: dict, folds: dict, seed: int = config.RANDOM_SEED) -> dict:
    """Orchestrator Decision Tree: baseline default (belum tuning), sama alurnya dengan Logistic Regression."""
    model_name = "decision_tree"
    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN

    model_factory = lambda: DecisionTreeClassifier(class_weight="balanced", random_state=seed)

    start_time = time.time()
    cv_result = run_stratified_cv(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor,
        model_factory=model_factory,
        threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor,
        model_factory=model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(split["development_ids"])][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json({"fold_metrics": cv_result["fold_metrics"]}, model_name, suffix="history")
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {
            "class_weight": "balanced",
            "random_state": seed,
            "max_depth": final_pipeline.named_steps["model"].get_depth(),
            "n_leaves": final_pipeline.named_steps["model"].get_n_leaves(),
            "feature_path": "tree",
        },
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name,
        "run_id": f"{model_name}_v1",
        "status": "done",
        "feature_path": "tree",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"],
        "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"],
        "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "training_time_seconds": training_time,
        "random_seed": seed,
        "model_path": str(pipeline_path),
        "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"],
        "cv_summary": cv_summary,
        "pipeline": final_pipeline,
        "oof_true": cv_result["oof_true"],
        "oof_proba": cv_result["oof_proba"],
    }

# Trainer Random Forest
# Sekali lagi, karena infrastruktur generik, orchestrator ini sangat mirip — beda model_factory saja (tambah n_estimators sebagai baseline 
# reasonable, bukan default sklearn yang cuma 100 pohon tanpa batas depth juga, tapi ensemble-nya sendiri yang meredam overfitting).

from sklearn.ensemble import RandomForestClassifier


def train_random_forest(df: pd.DataFrame, metadata: dict, split: dict, folds: dict, seed: int = config.RANDOM_SEED) -> dict:
    """Orchestrator Random Forest: baseline ensemble, alur sama dengan Logistic Regression/Decision Tree."""
    model_name = "random_forest"
    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN

    model_factory = lambda: RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1,
    )

    start_time = time.time()
    cv_result = run_stratified_cv(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor,
        model_factory=model_factory,
        threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor,
        model_factory=model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(split["development_ids"])][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json({"fold_metrics": cv_result["fold_metrics"]}, model_name, suffix="history")
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": seed,
            "oob_score": False,
            "feature_path": "tree",
        },
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name,
        "run_id": f"{model_name}_v1",
        "status": "done",
        "feature_path": "tree",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"],
        "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"],
        "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "training_time_seconds": training_time,
        "random_seed": seed,
        "model_path": str(pipeline_path),
        "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"],
        "cv_summary": cv_summary,
        "pipeline": final_pipeline,
        "oof_true": cv_result["oof_true"],
        "oof_proba": cv_result["oof_proba"],
    }

def _fit_and_evaluate_fold_lightgbm(
    model, preprocessor, X_train, y_train, X_val, y_val,
    early_stopping_rounds: int = 100, eval_metric: str = "auc", threshold: float = 0.5,
) -> dict:
    """Fit model boosting dengan eval_set (validation saja) + early stopping."""
    import lightgbm as lgb

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_val_transformed, y_val)],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(early_stopping_rounds, first_metric_only=True, verbose=False), lgb.log_evaluation(0)],
    )

    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
    metrics = evaluate_predictions(y_val, y_val_proba, threshold=threshold)
    metrics["y_val_true"] = np.asarray(y_val).tolist()
    metrics["y_val_proba"] = np.asarray(y_val_proba).tolist()

    best_iter = getattr(model, "best_iteration_", None)
    metrics["best_iteration"] = int(best_iter) if best_iter else model.n_estimators

    evals = model.evals_result_
    valid_key = list(evals.keys())[-1]  # ambil key eval set apa pun namanya, tidak di-hardcode
    metrics["iteration_history"] = {"valid_auc": evals[valid_key][eval_metric]}

    return metrics

# Loop CV Khusus Boosting
def run_stratified_cv_boosting(
    df: pd.DataFrame, folds: dict, feature_list: list, target_column: str,
    preprocessor_builder, model_factory, fold_fit_fn,
    id_column: str = config.ID_COLUMN,
    early_stopping_rounds: int = 100, eval_metric: str = "auc", threshold: float = 0.5,
) -> dict:
    """Loop CV untuk model boosting apa pun — fold_fit_fn menentukan cara fit spesifik librarynya."""
    dev_ids = np.array(folds["development_ids"])
    fold_arr = np.array(folds["fold_assignment"])
    n_splits = folds["n_splits"]

    dev_df = df.set_index(id_column).loc[dev_ids].reset_index()

    fold_metrics = []
    iteration_histories = []
    oof_true = np.full(len(dev_ids), fill_value=np.nan)
    oof_proba = np.full(len(dev_ids), fill_value=np.nan)

    for fold_idx in range(n_splits):
        train_mask = fold_arr != fold_idx
        val_mask = fold_arr == fold_idx

        X_train = dev_df.loc[train_mask, feature_list]
        y_train = dev_df.loc[train_mask, target_column].values
        X_val = dev_df.loc[val_mask, feature_list]
        y_val = dev_df.loc[val_mask, target_column].values

        preprocessor = preprocessor_builder(feature_list)
        model = model_factory()

        result = fold_fit_fn(
            model, preprocessor, X_train, y_train, X_val, y_val,
            early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, threshold=threshold,
        )

        iteration_histories.append(result.get("iteration_history"))
        fold_result = {k: v for k, v in result.items() if k not in ("y_val_true", "y_val_proba", "iteration_history")}
        fold_result["fold"] = fold_idx
        fold_metrics.append(fold_result)

        oof_true[val_mask] = result["y_val_true"]
        oof_proba[val_mask] = result["y_val_proba"]

    return {
        "fold_metrics": fold_metrics, "oof_true": oof_true, "oof_proba": oof_proba,
        "iteration_histories": iteration_histories,
    }

# Orchestrator train_lightgbm
# scale_pos_weight dihitung sekali dari label seluruh development set (bukan per-fold, karena stratifikasi 
# menjamin rasio kelas hampir identik di semua fold-train), lalu dibekukan sebagai hyperparameter tetap di 
# model_factory — menjaga signature tetap zero-argument sesuai kesepakatan.

def train_lightgbm(
    df: pd.DataFrame, metadata: dict, split: dict, folds: dict,
    seed: int = config.RANDOM_SEED, early_stopping_rounds: int = 100, eval_metric: str = "auc",
    model_name: str = "lightgbm",   # <- TAMBAHAN parameter baru
) -> dict:
    """Orchestrator LightGBM: CV dengan early stopping -> agregasi -> retrain -> simpan -> update summary."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "LightGBM belum terinstal. Jalankan `pip install lightgbm` di environment Homecreadit."
        ) from e

    # baris "model_name = 'lightgbm'" DIHAPUS -- sudah jadi parameter
    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN

    dev_ids = split["development_ids"]
    y_dev = df[df[config.ID_COLUMN].isin(dev_ids)][target_column].values
    spw = compute_scale_pos_weight(y_dev)

    model_factory = lambda: lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, verbosity=-1, metric=eval_metric,
    )

    start_time = time.time()
    cv_result = run_stratified_cv_boosting(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor, model_factory=model_factory,
        fold_fit_fn=_fit_and_evaluate_fold_lightgbm,   # <- TAMBAHKAN baris ini
        early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    best_iteration_mean = int(round(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])))

    final_model_factory = lambda: lgb.LGBMClassifier(
        n_estimators=best_iteration_mean, learning_rate=0.05, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, verbosity=-1, metric=eval_metric,
    )

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor, model_factory=final_model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(dev_ids)][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json(
        {"fold_metrics": cv_result["fold_metrics"], "iteration_history_per_fold": cv_result["iteration_histories"]},
        model_name, suffix="history",
    )
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {
            "n_estimators": 1000, "learning_rate": 0.05, "scale_pos_weight": spw,
            "early_stopping_rounds": early_stopping_rounds, "eval_metric": eval_metric,
            "random_state": seed, "feature_path": "tree", "encoding": "one_hot",
        },
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name, "run_id": f"{model_name}_v1", "status": "done",
        "feature_path": "tree",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"], "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"], "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "best_iteration_mean": float(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])),
        "training_time_seconds": training_time, "random_seed": seed,
        "model_path": str(pipeline_path), "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"], "cv_summary": cv_summary,
        "pipeline": final_pipeline, "oof_true": cv_result["oof_true"], "oof_proba": cv_result["oof_proba"],
    }

def _fit_and_evaluate_fold_xgboost(
    model, preprocessor, X_train, y_train, X_val, y_val,
    early_stopping_rounds: int = 100, eval_metric: str = "auc", threshold: float = 0.5,
) -> dict:
    """Fit XGBoost dengan eval_set + early stopping (early_stopping_rounds sudah di constructor model)."""
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    model.fit(X_train_transformed, y_train, eval_set=[(X_val_transformed, y_val)], verbose=False)

    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
    metrics = evaluate_predictions(y_val, y_val_proba, threshold=threshold)
    metrics["y_val_true"] = np.asarray(y_val).tolist()
    metrics["y_val_proba"] = np.asarray(y_val_proba).tolist()

    best_iter = getattr(model, "best_iteration", None)
    metrics["best_iteration"] = int(best_iter) if best_iter is not None else model.n_estimators

    evals = model.evals_result()
    valid_key = list(evals.keys())[-1]
    metric_key = list(evals[valid_key].keys())[-1]
    metrics["iteration_history"] = {"valid_auc": evals[valid_key][metric_key]}

    return metrics


def train_xgboost(
    df: pd.DataFrame, metadata: dict, split: dict, folds: dict,
    seed: int = config.RANDOM_SEED, early_stopping_rounds: int = 100, eval_metric: str = "auc",
    model_name: str = "xgboost",   # <- TAMBAHAN parameter baru
) -> dict:
    """Orchestrator XGBoost: CV dengan early stopping -> agregasi -> retrain -> simpan -> update summary."""
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("XGBoost belum terinstal. Jalankan `pip install xgboost`.") from e

    # baris "model_name = 'xgboost'" DIHAPUS -- sudah jadi parameter
    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN

    dev_ids = split["development_ids"]
    y_dev = df[df[config.ID_COLUMN].isin(dev_ids)][target_column].values
    spw = compute_scale_pos_weight(y_dev)

    model_factory = lambda: xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, eval_metric=eval_metric,
        early_stopping_rounds=early_stopping_rounds,
    )

    start_time = time.time()
    cv_result = run_stratified_cv_boosting(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor, model_factory=model_factory,
        fold_fit_fn=_fit_and_evaluate_fold_xgboost,
        early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    best_iteration_mean = int(round(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])))

    final_model_factory = lambda: xgb.XGBClassifier(
        n_estimators=best_iteration_mean, learning_rate=0.05, scale_pos_weight=spw,
        random_state=seed, n_jobs=-1, eval_metric=eval_metric,
    )  # TANPA early_stopping_rounds — tidak ada eval_set saat retrain di seluruh dev set

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_tree_preprocessor, model_factory=final_model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(dev_ids)][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json(
        {"fold_metrics": cv_result["fold_metrics"], "iteration_history_per_fold": cv_result["iteration_histories"]},
        model_name, suffix="history",
    )
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {
            "n_estimators": 1000, "learning_rate": 0.05, "scale_pos_weight": spw,
            "early_stopping_rounds": early_stopping_rounds, "eval_metric": eval_metric,
            "random_state": seed, "feature_path": "tree", "encoding": "one_hot",
        },
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name, "run_id": f"{model_name}_v1", "status": "done",
        "feature_path": "tree",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"], "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"], "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "best_iteration_mean": float(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])),
        "training_time_seconds": training_time, "random_seed": seed,
        "model_path": str(pipeline_path), "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"], "cv_summary": cv_summary,
        "pipeline": final_pipeline, "oof_true": cv_result["oof_true"], "oof_proba": cv_result["oof_proba"],
    }




class CatBoostColumnSelector(BaseEstimator, TransformerMixin):
    """
    Memilih kolom feature_list tanpa mengubah nama/tipe data — dipakai CatBoost
    karena kategorikal dikirim mentah (tanpa One-Hot). Dibuat sebagai class
    (bukan lambda) supaya bisa di-pickle oleh joblib saat menyimpan pipeline.
    """

    def __init__(self, feature_list: list):
        self.feature_list = feature_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_list].copy()


def build_catboost_preprocessor(feature_list: list) -> CatBoostColumnSelector:
    """CatBoost menerima fitur kategorikal MENTAH (bukan One-Hot), tanpa scaler."""
    return CatBoostColumnSelector(feature_list)

def _fit_and_evaluate_fold_catboost(
    model, preprocessor, X_train, y_train, X_val, y_val,
    early_stopping_rounds: int = 100, eval_metric: str = "AUC", threshold: float = 0.5,
) -> dict:
    """Fit CatBoost (cat_features sudah di constructor model) dengan eval_set + early stopping."""
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    model.fit(
        X_train_transformed, y_train,
        eval_set=(X_val_transformed, y_val),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
    metrics = evaluate_predictions(y_val, y_val_proba, threshold=threshold)
    metrics["y_val_true"] = np.asarray(y_val).tolist()
    metrics["y_val_proba"] = np.asarray(y_val_proba).tolist()

    best_iter = model.get_best_iteration()
    metrics["best_iteration"] = int(best_iter) if best_iter is not None else model.tree_count_

    evals = model.get_evals_result()
    valid_key = list(evals.keys())[-1]
    metric_key = list(evals[valid_key].keys())[-1]
    metrics["iteration_history"] = {"valid_auc": evals[valid_key][metric_key]}

    return metrics

def train_catboost(
    df: pd.DataFrame, metadata: dict, split: dict, folds: dict,
    seed: int = config.RANDOM_SEED, early_stopping_rounds: int = 100, eval_metric: str = "AUC",
    model_name: str = "catboost",
    extra_params: dict = None,   # <- TAMBAHAN: hyperparameter hasil tuning (depth, l2_leaf_reg, dst)
) -> dict:
    """Orchestrator CatBoost: kategorikal RAW (bukan One-Hot), CV early stopping -> retrain -> simpan -> summary."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError("CatBoost belum terinstal. Jalankan `pip install catboost`.") from e

    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN
    categorical_cols, _ = _split_categorical_numeric(feature_list)
    extra_params = extra_params or {}

    dev_ids = split["development_ids"]
    y_dev = df[df[config.ID_COLUMN].isin(dev_ids)][target_column].values
    spw = compute_scale_pos_weight(y_dev)

    base_params = {
        "iterations": 1000, "learning_rate": 0.05, "scale_pos_weight": spw,
        "random_state": seed, "thread_count": -1, "verbose": False,
        "eval_metric": eval_metric, "cat_features": categorical_cols,
    }
    base_params.update(extra_params)   # <- override default kalau ada di hasil tuning
    model_factory = lambda: CatBoostClassifier(**base_params)

    start_time = time.time()
    cv_result = run_stratified_cv_boosting(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_catboost_preprocessor, model_factory=model_factory,
        fold_fit_fn=_fit_and_evaluate_fold_catboost,
        early_stopping_rounds=early_stopping_rounds, eval_metric=eval_metric, threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    best_iteration_mean = int(round(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])))

    final_params = dict(base_params)
    final_params["iterations"] = best_iteration_mean
    final_model_factory = lambda: CatBoostClassifier(**final_params)

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_catboost_preprocessor, model_factory=final_model_factory,
    )

    dev_sample = df[df[config.ID_COLUMN].isin(dev_ids)][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json(
        {"fold_metrics": cv_result["fold_metrics"], "iteration_history_per_fold": cv_result["iteration_histories"]},
        model_name, suffix="history",
    )
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {**base_params, "cat_features": categorical_cols, "feature_path": "tree", "encoding": "raw_categorical"},
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name, "run_id": f"{model_name}_v1", "status": "done",
        "feature_path": "tree",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"], "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"], "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "best_iteration_mean": float(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])),
        "training_time_seconds": training_time, "random_seed": seed,
        "model_path": str(pipeline_path), "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"], "cv_summary": cv_summary,
        "pipeline": final_pipeline, "oof_true": cv_result["oof_true"], "oof_proba": cv_result["oof_proba"],
    }

def tune_catboost_hyperparameters(
    df: pd.DataFrame, metadata: dict, split: dict,
    n_trials: int = 25, seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Mencari hyperparameter CatBoost terbaik pakai Optuna. Tiap trial dievaluasi
    dengan SATU split train/validation (80/20), bukan full 5-fold CV -- supaya
    cepat. Hasil terbaiknya divalidasi ulang dengan CV penuh lewat train_catboost().
    """
    import optuna
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split as tts

    feature_list = metadata["tree_features"]
    target_column = config.TARGET_COLUMN
    categorical_cols, _ = _split_categorical_numeric(feature_list)

    dev_ids = split["development_ids"]
    dev_df = df[df[config.ID_COLUMN].isin(dev_ids)]
    X = dev_df[feature_list]
    y = dev_df[target_column].values

    X_train, X_val, y_train, y_val = tts(X, y, test_size=0.2, stratify=y, random_state=seed)
    spw = compute_scale_pos_weight(y_train)

    def objective(trial):
        params = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "scale_pos_weight": spw,
            "random_state": seed,
            "thread_count": -1,
            "verbose": False,
            "eval_metric": "AUC",
            "cat_features": categorical_cols,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, verbose=False)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_val_proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {"best_params": study.best_params, "best_value": study.best_value, "study": study}


#MLP
def retrain_on_full_development(
    df: pd.DataFrame,
    split: dict,
    feature_list: list,
    target_column: str,
    preprocessor_builder,
    model_factory,
    id_column: str = config.ID_COLUMN,
    fit_params: dict = None,   # <- TAMBAHAN: default None, tidak mengubah perilaku model lain
) -> Pipeline:
    """Fit satu Pipeline (preprocessor+model) pada SELURUH development set. Dipanggil setelah CV selesai."""
    dev_ids = split["development_ids"]
    dev_df = df[df[id_column].isin(dev_ids)]

    X_dev = dev_df[feature_list]
    y_dev = dev_df[target_column].values

    pipeline = Pipeline([
        ("preprocessor", preprocessor_builder(feature_list)),
        ("model", model_factory()),
    ])
    pipeline.fit(X_dev, y_dev, **(fit_params or {}))   # <- UBAH: dari pipeline.fit(X_dev, y_dev)
    return pipeline

def _fit_and_evaluate_fold_mlp(
    model, preprocessor, X_train, y_train, X_val, y_val,
    early_stopping_rounds: int = 100, eval_metric: str = "auc", threshold: float = 0.5,
) -> dict:
    """
    Fit MLP dengan sample_weight (pengganti class_weight yang tidak didukung MLPClassifier).
    early_stopping_rounds & eval_metric tidak dipakai di sini -- MLPClassifier menangani
    early stopping sendiri (validation_fraction internal dari data training). Parameter ini
    hanya ada supaya signature kompatibel dengan run_stratified_cv_boosting.
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train_transformed, y_train, sample_weight=sample_weight)

    y_val_proba = model.predict_proba(X_val_transformed)[:, 1]
    metrics = evaluate_predictions(y_val, y_val_proba, threshold=threshold)
    metrics["y_val_true"] = np.asarray(y_val).tolist()
    metrics["y_val_proba"] = np.asarray(y_val_proba).tolist()

    metrics["best_iteration"] = int(model.n_iter_)
    metrics["iteration_history"] = {"loss_curve": model.loss_curve_}

    return metrics

def train_mlp(
    df: pd.DataFrame, metadata: dict, split: dict, folds: dict,
    seed: int = config.RANDOM_SEED,
) -> dict:
    """
    Orchestrator MLP (Opsi A): MLPClassifier + sample_weight manual, karena MLPClassifier
    tidak mendukung class_weight="balanced" seperti model sklearn lainnya. Reuse
    build_linear_mlp_preprocessor (One-Hot + StandardScaler) & run_stratified_cv_boosting
    (fold_fit_fn generik, meski MLP bukan model boosting).
    """
    from sklearn.neural_network import MLPClassifier

    model_name = "mlp"
    feature_list = metadata["linear_mlp_features"]
    target_column = config.TARGET_COLUMN

    model_factory = lambda: MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=1e-3, max_iter=200,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        random_state=seed,
    )

    start_time = time.time()
    cv_result = run_stratified_cv_boosting(
        df, folds, feature_list, target_column,
        preprocessor_builder=build_linear_mlp_preprocessor, model_factory=model_factory,
        fold_fit_fn=_fit_and_evaluate_fold_mlp,
        early_stopping_rounds=100, eval_metric="auc", threshold=0.5,
    )
    cv_summary = aggregate_cv_metrics(cv_result["fold_metrics"])
    training_time = time.time() - start_time

    dev_ids = split["development_ids"]
    y_dev_full = df[df[config.ID_COLUMN].isin(dev_ids)][target_column].values
    sample_weight_full = compute_sample_weight(class_weight="balanced", y=y_dev_full)

    final_pipeline = retrain_on_full_development(
        df, split, feature_list, target_column,
        preprocessor_builder=build_linear_mlp_preprocessor, model_factory=model_factory,
        fit_params={"model__sample_weight": sample_weight_full},
    )

    dev_sample = df[df[config.ID_COLUMN].isin(dev_ids)][feature_list].head(5)
    n_features_after = final_pipeline.named_steps["preprocessor"].transform(dev_sample).shape[1]

    pipeline_path = save_model_pipeline(final_pipeline, model_name)
    history_path = save_model_json(
        {"fold_metrics": cv_result["fold_metrics"], "iteration_history_per_fold": cv_result["iteration_histories"]},
        model_name, suffix="history",
    )
    save_model_json(cv_summary, model_name, suffix="metrics")
    save_model_json(
        {
            "hidden_layer_sizes": [64, 32], "activation": "relu", "solver": "adam",
            "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 200,
            "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 10,
            "random_state": seed, "feature_path": "linear_mlp",
            "imbalance_handling": "sample_weight (compute_sample_weight balanced)",
        },
        model_name, suffix="config",
    )

    update_model_summary({
        "model_name": model_name, "run_id": f"{model_name}_v1", "status": "done",
        "feature_path": "linear_mlp",
        "n_features_before_encoding": len(feature_list),
        "n_features_after_encoding": n_features_after,
        "cv_roc_auc_mean": cv_summary["roc_auc_mean"], "cv_roc_auc_std": cv_summary["roc_auc_std"],
        "cv_pr_auc_mean": cv_summary["pr_auc_mean"], "cv_pr_auc_std": cv_summary["pr_auc_std"],
        "best_iteration_mean": float(np.mean([fm["best_iteration"] for fm in cv_result["fold_metrics"]])),
        "training_time_seconds": training_time, "random_seed": seed,
        "model_path": str(pipeline_path), "history_path": str(history_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    return {
        "cv_fold_metrics": cv_result["fold_metrics"], "cv_summary": cv_summary,
        "pipeline": final_pipeline, "oof_true": cv_result["oof_true"], "oof_proba": cv_result["oof_proba"],
    }

# ===================================================================
# Perbandingan & Ranking Model (Laporan Ringkas untuk Notebook)
# ===================================================================

def display_model_comparison(summary_path: Path = config.MODEL_SUMMARY_PATH):
    """
    Menampilkan tabel ranking perbandingan seluruh model langsung di notebook --
    hanya kolom penting (bukan path file mentah), diurutkan dari CV ROC-AUC
    tertinggi, dengan format angka rapi dan highlight warna pada model terbaik.
    """
    columns = [
        "model_name", "feature_path", "n_features_after_encoding",
        "cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean", "cv_pr_auc_std",
        "best_iteration_mean", "training_time_seconds",
    ]
    summary = pd.read_csv(summary_path)
    comparison = summary[columns].sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    comparison.insert(0, "rank", comparison.index + 1)

    rename_map = {
        "model_name": "Model", "feature_path": "Jalur Fitur",
        "n_features_after_encoding": "N Fitur", "cv_roc_auc_mean": "ROC-AUC (mean)",
        "cv_roc_auc_std": "ROC-AUC (std)", "cv_pr_auc_mean": "PR-AUC (mean)",
        "cv_pr_auc_std": "PR-AUC (std)", "best_iteration_mean": "Best Iter",
        "training_time_seconds": "Waktu Training (s)",
    }
    comparison = comparison.rename(columns=rename_map).set_index("rank")

    styled = (
        comparison.style
        .format({
            "ROC-AUC (mean)": "{:.4f}", "ROC-AUC (std)": "{:.4f}",
            "PR-AUC (mean)": "{:.4f}", "PR-AUC (std)": "{:.4f}",
            "Best Iter": "{:.1f}", "Waktu Training (s)": "{:.1f}",
        }, na_rep="-")
        .background_gradient(subset=["ROC-AUC (mean)", "PR-AUC (mean)"], cmap="Greens")
        .set_properties(**{"text-align": "center"})
        .set_caption("Perbandingan CV Performa Seluruh Model — Home Credit Default Risk")
    )
    return styled

# Stacking di sini artinya: melatih model kedua (Logistic Regression) yang belajar sendiri bobot optimal dari 
# 3 prediksi OOF (CatBoost, LightGBM, XGBoost) — bukan kita tebak manual (50/30/20) seperti sebelumnya. 
# Supaya hasilnya jujur (tidak ada kebocoran data), meta-model ini dievaluasi memakai fold yang sama 
# persis dengan yang dipakai model dasar: di-fit hanya di fold-train, diuji di fold-validation, per fold 
# — sama seperti model dasar, bukan langsung dilatih dan diuji di data yang sama.
def train_stacking_meta_model(
    oof_true: np.ndarray, oof_proba_dict: dict, folds: dict, threshold: float = 0.5,
) -> dict:
    """
    Melatih meta-model (Logistic Regression) di atas OOF prediksi beberapa model
    dasar (stacking level-2). Dievaluasi dengan fold_assignment yang SAMA seperti
    model dasar -- fit hanya di fold-train, uji di fold-validation, supaya tidak
    ada kebocoran data di level meta-model.
    """
    from sklearn.linear_model import LogisticRegression

    model_names = list(oof_proba_dict.keys())
    X_meta = np.column_stack([oof_proba_dict[name] for name in model_names])
    fold_arr = np.array(folds["fold_assignment"])
    n_splits = folds["n_splits"]

    meta_oof_proba = np.full(len(oof_true), fill_value=np.nan)
    fold_metrics = []
    fold_coefficients = []

    for fold_idx in range(n_splits):
        train_mask = fold_arr != fold_idx
        val_mask = fold_arr == fold_idx

        meta_model = LogisticRegression(class_weight="balanced")
        meta_model.fit(X_meta[train_mask], oof_true[train_mask])
        val_proba = meta_model.predict_proba(X_meta[val_mask])[:, 1]
        meta_oof_proba[val_mask] = val_proba

        fold_result = evaluate_predictions(oof_true[val_mask], val_proba, threshold=threshold)
        fold_result["fold"] = fold_idx
        fold_metrics.append(fold_result)
        fold_coefficients.append(dict(zip(model_names, meta_model.coef_[0].tolist())))

    cv_summary = aggregate_cv_metrics(fold_metrics)
    overall_metrics = evaluate_predictions(oof_true, meta_oof_proba, threshold=threshold)

    final_meta_model = LogisticRegression(class_weight="balanced")
    final_meta_model.fit(X_meta, oof_true)

    return {
        "cv_summary": cv_summary,
        "fold_metrics": fold_metrics,
        "overall_oof_metrics": overall_metrics,
        "meta_oof_proba": meta_oof_proba,
        "model_names_order": model_names,
        "fold_coefficients": fold_coefficients,
        "final_meta_model": final_meta_model,
    }

def save_stacking_meta_model(stacking_result: dict, model_name: str = "stacking_meta") -> dict:
    """Menyimpan meta-model final dan metadata koefisien ke models/{model_name}/."""
    model_dir = get_model_dir(model_name)
    model_path = model_dir / f"{model_name}_model.joblib"
    joblib.dump(stacking_result["final_meta_model"], model_path)

    metrics_path = save_model_json(stacking_result["cv_summary"], model_name, suffix="metrics")
    config_path = save_model_json(
        {
            "model_names_order": stacking_result["model_names_order"],
            "fold_coefficients": stacking_result["fold_coefficients"],
            "final_coefficients": dict(zip(
                stacking_result["model_names_order"],
                stacking_result["final_meta_model"].coef_[0].tolist(),
            )),
            "final_intercept": float(stacking_result["final_meta_model"].intercept_[0]),
        },
        model_name, suffix="config",
    )
    return {"model_path": model_path, "metrics_path": metrics_path, "config_path": config_path}


def save_oof_predictions(oof_true: np.ndarray, oof_proba: np.ndarray, model_name: str) -> Path:
    """Menyimpan oof_true dan oof_proba ke CSV supaya bisa dimuat ulang di notebook lain."""
    model_dir = get_model_dir(model_name)
    oof_path = model_dir / f"{model_name}_oof.csv"
    pd.DataFrame({"oof_true": oof_true, "oof_proba": oof_proba}).to_csv(oof_path, index=False)
    return oof_path