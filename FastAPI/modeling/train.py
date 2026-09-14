"""
Versi minimal train.py untuk deployment API.
Hanya berisi class CatBoostColumnSelector yang dibutuhkan joblib.load()
untuk deserialize pipeline model.

CATATAN: Path modul ini SENGAJA "modeling.train" (bukan "src.modeling.train")
karena saat model aslinya disimpan (joblib.dump di notebook training), class
ini dikenali Python dengan path "modeling.train" — kemungkinan karena folder
src/ ditambahkan langsung ke sys.path saat training, bukan diinstall sebagai
package "src". joblib/pickle membekukan path modul persis seperti itu, jadi
struktur folder ini WAJIB mengikuti path asli tersebut, bukan yang dipakai
di prediction_service.py untuk import biasa.
"""

from sklearn.base import BaseEstimator, TransformerMixin


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