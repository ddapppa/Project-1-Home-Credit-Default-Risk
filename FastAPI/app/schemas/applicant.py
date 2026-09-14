"""
Skema Pydantic untuk endpoint /predict.

Ada 4 kelas dengan peran berbeda:
1. ApplicantModeExisting -> input untuk Mode Existing (lookup histori by SK_ID_CURR)
2. ApplicantModeNew      -> input untuk Mode Baru (form manual, SATU-SATUNYA yang masuk ke model)
3. ApplicationContext    -> field UI-only, TIDAK PERNAH dikirim ke pipeline.predict()
4. PredictionResponse    -> bentuk response akhir ke frontend
"""

from datetime import date, timedelta
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Konstanta kategori — HARUS sama persis dengan nilai unik hasil df[col].unique()
# di data training. Jangan diubah manual tanpa mengecek ulang ke dataset.
# ---------------------------------------------------------------------------
NameIncomeType = Literal["Commercial associate", "Pensioner", "Rare", "State servant", "Working"]
NameEducationType = Literal[
    "Academic degree", "Higher education", "Incomplete higher",
    "Lower secondary", "Secondary / secondary special",
]
NameFamilyStatus = Literal["Civil marriage", "Married", "Separated", "Single / not married", "Widow"]
NameHousingType = Literal[
    "Co-op apartment", "House / apartment", "Municipal apartment",
    "Office apartment", "Rented apartment", "With parents",
]
NameTypeSuite = Literal[
    "Children", "Family", "Group of people", "Other_A", "Other_B",
    "Spouse, partner", "Unaccompanied",
]
OccupationType = Literal[
    "Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers",
    "HR staff", "High skill tech staff", "IT staff", "Laborers",
    "Low-skill Laborers", "Managers", "Medicine staff", "Private service staff",
    "Realty agents", "Sales staff", "Secretaries", "Security staff",
    "Unknown_Occupation", "Waiters/barmen staff",
]
OrganizationType = Literal[
    "Advertising", "Agriculture", "Bank", "Business Entity Type 1", "Business Entity Type 2",
    "Business Entity Type 3", "Cleaning", "Construction", "Culture", "Electricity", "Emergency",
    "Government", "Hotel", "Housing", "Industry: type 1", "Industry: type 10", "Industry: type 11",
    "Industry: type 12", "Industry: type 13", "Industry: type 2", "Industry: type 3",
    "Industry: type 4", "Industry: type 5", "Industry: type 6", "Industry: type 7",
    "Industry: type 9", "Insurance", "Kindergarten", "Legal Services", "Medicine", "Military",
    "Mobile", "Not_Working", "Other", "Police", "Postal", "Rare", "Realtor", "Religion",
    "Restaurant", "School", "Security", "Security Ministries", "Self-employed", "Services",
    "Telecom", "Trade: type 1", "Trade: type 2", "Trade: type 3", "Trade: type 4", "Trade: type 6",
    "Trade: type 7", "Transport: type 1", "Transport: type 2", "Transport: type 3",
    "Transport: type 4", "University",
]


# ---------------------------------------------------------------------------
# Aturan batas usia pengajuan (Opsi C — dikonfirmasi bersama pengguna):
#
# - Status "Single / not married": minimal 21 tahun.
#   Dasar: KUHPerdata Pasal 330 — batas umum kecakapan hukum (dewasa) adalah
#   21 tahun bagi yang belum pernah menikah.
#
# - Status "Married", "Civil marriage", "Separated", "Widow": minimal 19 tahun.
#   Dasar: (a) KUHPerdata Pasal 330 — seseorang yang SUDAH/PERNAH menikah
#   dianggap cakap hukum meski belum 21 tahun; (b) UU No. 16 Tahun 2019
#   (revisi UU Perkawinan) menetapkan usia minimal sah menikah 19 tahun
#   untuk pria maupun wanita. "Separated" dan "Widow" tetap dihitung pernah
#   menikah, sehingga status cakap hukumnya tidak hilang.
#
# Batas atas usia (100 tahun) tetap berlaku untuk semua status.
# ---------------------------------------------------------------------------
MIN_AGE_GENERAL = 21
MIN_AGE_MARRIED = 19
MAX_AGE = 100
EVER_MARRIED_STATUSES = {"Married", "Civil marriage", "Separated", "Widow"}


# ---------------------------------------------------------------------------
# Konstanta batas bisnis — mudah diubah tanpa menyentuh logika validasi
# ---------------------------------------------------------------------------
AMT_INCOME_TOTAL_MIN = 1_500_000   # acuan: UMP terendah nasional (konservatif)
AMT_CREDIT_MIN = 1_000_000          # batas bawah kredit realistis (KTA/kredit barang)
TENOR_MIN_BULAN = 1                  # minimal 1x angsuran (bukan lunas sekaligus di muka)
TENOR_MAX_BULAN = 120                # maksimal 10 tahun, wajar untuk kredit konsumtif


# ---------------------------------------------------------------------------
# 1. Mode Existing — hanya butuh SK_ID_CURR untuk lookup histori lengkap
# ---------------------------------------------------------------------------
class ApplicantModeExisting(BaseModel):
    sk_id_curr: int = Field(..., description="ID nasabah existing dari holdout set (test_ids)")


# ---------------------------------------------------------------------------
# 2. Mode Baru — SATU-SATUNYA skema yang datanya diteruskan ke pipeline.predict()
# ---------------------------------------------------------------------------
class ApplicantModeNew(BaseModel):
    # --- Kategorikal ---
    name_contract_type: Literal["Cash loans", "Revolving loans"]
    code_gender: Literal["M", "F"]
    flag_own_car: Literal["Y", "N"]
    flag_own_realty: Literal["Y", "N"]
    name_type_suite: NameTypeSuite
    name_income_type: NameIncomeType
    name_education_type: NameEducationType
    name_family_status: NameFamilyStatus
    name_housing_type: NameHousingType

    # --- Kontrol status pekerjaan (menentukan occupation/organization otomatis) ---
    status_pekerjaan: Literal["Bekerja", "Tidak Bekerja"]
    occupation_type: Optional[OccupationType] = None
    organization_type: Optional[OrganizationType] = None
    tanggal_mulai_kerja: Optional[date] = None

    # --- Numerik mentah ---
    cnt_children: int = Field(ge=0, le=20, description="Jumlah anak")
    cnt_fam_members: int = Field(ge=1, le=20, description="Jumlah anggota keluarga")
    amt_income_total: float = Field(
        ge=AMT_INCOME_TOTAL_MIN,
        description="Total Penghasilan per Bulan (Rp)"
    )
    amt_credit: float = Field(
        ge=AMT_CREDIT_MIN,
        description="Total Kredit yang Diajukan (Rp)"
    )
    amt_annuity: float = Field(
        gt=0,
        description="Angsuran per Bulan (Rp)"
    )
    amt_goods_price: Optional[float] = Field(None, description="Harga barang (khusus consumer loan)")
    own_car_age: Optional[float] = Field(None, ge=0, le=80, description="Usia mobil, kosongkan jika tidak punya")

    # --- Tanggal (dikonversi backend jadi DAYS_BIRTH / DAYS_EMPLOYED) ---
    tanggal_lahir: date

    # -----------------------------------------------------------------------
    # Validator 1: sinkronisasi status_pekerjaan dengan occupation/organization
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def sync_status_pekerjaan(self):
        if self.status_pekerjaan == "Tidak Bekerja":
            self.occupation_type = "Unknown_Occupation"
            self.organization_type = "Not_Working"
            self.tanggal_mulai_kerja = None
        else:
            if self.occupation_type is None or self.organization_type is None:
                raise ValueError(
                    "occupation_type dan organization_type wajib diisi jika status_pekerjaan = 'Bekerja'"
                )
            if self.tanggal_mulai_kerja is None:
                raise ValueError(
                    "tanggal_mulai_kerja wajib diisi jika status_pekerjaan = 'Bekerja'"
                )
        return self

    # -----------------------------------------------------------------------
    # Validator 2: usia wajar, tergantung status pernikahan (Opsi C)
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_tanggal_lahir(self):
        today = date.today()
        if self.tanggal_lahir > today:
            raise ValueError("tanggal_lahir tidak boleh di masa depan")

        usia_hari = (today - self.tanggal_lahir).days
        usia_tahun = usia_hari / 365.25

        if usia_tahun > MAX_AGE:
            raise ValueError(f"Usia nasabah tidak wajar ({usia_tahun:.1f} tahun). Maksimal {MAX_AGE} tahun.")

        is_ever_married = self.name_family_status in EVER_MARRIED_STATUSES
        usia_minimal = MIN_AGE_MARRIED if is_ever_married else MIN_AGE_GENERAL

        if usia_tahun < usia_minimal:
            if is_ever_married:
                raise ValueError(
                    f"Usia nasabah ({usia_tahun:.1f} tahun) di bawah batas minimal {usia_minimal} tahun "
                    f"untuk status pernikahan '{self.name_family_status}' "
                    "(sesuai UU No. 16/2019 tentang Perkawinan, usia minimal sah menikah 19 tahun)."
                )
            raise ValueError(
                f"Usia nasabah ({usia_tahun:.1f} tahun) di bawah batas minimal {usia_minimal} tahun "
                "untuk pemohon berstatus belum menikah "
                "(sesuai KUHPerdata Pasal 330, batas dewasa umum 21 tahun)."
            )
        return self

    # -----------------------------------------------------------------------
    # Validator 3: tanggal mulai kerja harus setelah usia minimal kerja (15 tahun)
    # dan tidak boleh di masa depan
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_tanggal_mulai_kerja(self):
        if self.tanggal_mulai_kerja is None:
            return self

        today = date.today()
        if self.tanggal_mulai_kerja > today:
            raise ValueError("tanggal_mulai_kerja tidak boleh di masa depan")

        usia_minimal_kerja = self.tanggal_lahir + timedelta(days=15 * 365)
        if self.tanggal_mulai_kerja < usia_minimal_kerja:
            raise ValueError("tanggal_mulai_kerja tidak wajar — nasabah belum berusia 15 tahun saat itu")
        return self

    # -----------------------------------------------------------------------
    # Validator 4: konsistensi jumlah anggota keluarga vs jumlah anak
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_family_composition(self):
        minimal_fam_members = self.cnt_children + 1  # diri sendiri + anak yang ditanggung
        if self.cnt_fam_members < minimal_fam_members:
            raise ValueError(
                f"Jumlah Anggota Keluarga ({self.cnt_fam_members}) tidak konsisten dengan "
                f"Jumlah Anak ({self.cnt_children}). Minimal harus {minimal_fam_members} "
                "(diri sendiri + anak yang menjadi tanggungan)."
            )
        return self

    # -----------------------------------------------------------------------
    # Validator 5: tenor implisit (rasio kredit / angsuran) harus wajar
    # -----------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_tenor_implisit(self):
        tenor_bulan = self.amt_credit / self.amt_annuity
        if tenor_bulan < TENOR_MIN_BULAN:
            raise ValueError(
                f"Kombinasi kredit dan angsuran tidak wajar: tenor implisit hanya "
                f"{tenor_bulan:.1f} bulan (angsuran terlalu besar dibanding total kredit, "
                f"minimal {TENOR_MIN_BULAN} bulan)."
            )
        if tenor_bulan > TENOR_MAX_BULAN:
            raise ValueError(
                f"Kombinasi kredit dan angsuran tidak wajar: tenor implisit mencapai "
                f"{tenor_bulan:.0f} bulan (angsuran terlalu kecil dibanding total kredit, "
                f"maksimal {TENOR_MAX_BULAN} bulan / 10 tahun)."
            )
        return self

# ---------------------------------------------------------------------------
# 3. UI-only — ditampilkan di halaman hasil, TIDAK PERNAH masuk ke model
# ---------------------------------------------------------------------------
class ApplicationContext(BaseModel):
    tujuan_dana: Optional[str] = Field(None, description="UI-only, tidak dipakai model")
    rencana_pembayaran: Optional[str] = Field(None, description="UI-only, tidak dipakai model")
    catatan_reviewer: Optional[str] = Field(None, description="UI-only, tidak dipakai model")


# ---------------------------------------------------------------------------
# 4. Response akhir ke frontend
# ---------------------------------------------------------------------------

class ReasonItem(BaseModel):
    feature: str = Field(..., description="Nama teknis fitur, untuk traceability/audit")
    label: str = Field(..., description="Label netral yang mudah dipahami admin")
    value_display: str = Field(..., description="Nilai aktual fitur untuk nasabah ini, sudah diformat")
    direction: Literal["menaikkan_risiko", "menurunkan_risiko"]
    contribution_pct: float
    
class PredictionResponse(BaseModel):
    mode: Literal["existing", "new"]
    probability_default: float = Field(..., ge=0.0, le=1.0)
    threshold: float = 0.6669
    keputusan: Literal["APPROVE", "REVIEW/REJECT"]
    disclaimer: Optional[str] = None
    reasons: Optional[List[ReasonItem]] = None  # <- BARU, hanya terisi untuk Mode Existing
    context: Optional[ApplicationContext] = None

class NewApplicantRequest(BaseModel):
    applicant: ApplicantModeNew
    context: Optional[ApplicationContext] = None