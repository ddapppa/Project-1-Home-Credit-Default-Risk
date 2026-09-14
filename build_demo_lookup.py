# build_demo_lookup.py
# Jalankan dari ROOT project Data Science (bukan dari folder FastAPI/):
#   cd D:\Kerjaan\Home Creadit\Project-1-Home-Credit-Default-Risk-main
#   conda activate Homecreadit
#   python build_demo_lookup.py

import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config
from modeling.train import load_featured_dataset, create_or_load_holdout_split

print("Loading application_train_featured_v3.csv ...")
df_v3 = load_featured_dataset(path=config.PROCESSED_DATA_DIR / "application_train_featured_v3.csv")

with open(config.PROCESSED_DATA_DIR / "feature_engineering_metadata_v3.json") as f:
    metadata_v3 = json.load(f)
feature_cols = metadata_v3["tree_features_v3"]

print("Loading holdout_split.json (test_ids) ...")
split = create_or_load_holdout_split(df_v3)
holdout_df = df_v3[df_v3[config.ID_COLUMN].isin(split["test_ids"])].reset_index(drop=True)

output_cols = [config.ID_COLUMN] + feature_cols
demo_lookup_df = holdout_df[output_cols].copy()

output_path = PROJECT_ROOT / "FastAPI" / "app" / "data" / "demo_lookup.parquet"
output_path.parent.mkdir(parents=True, exist_ok=True)
demo_lookup_df.to_parquet(output_path, index=False)

print(f"Selesai: {len(demo_lookup_df)} baris, {len(output_cols)} kolom")
print(f"Disimpan ke: {output_path}")
print(f"Contoh SK_ID_CURR yang bisa dites: {demo_lookup_df[config.ID_COLUMN].head(5).tolist()}")