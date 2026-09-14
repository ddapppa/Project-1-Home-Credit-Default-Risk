# tests/test_applicant_schema.py
from datetime import date
import pytest
from pydantic import ValidationError
from app.schemas.applicant import ApplicantModeNew


@pytest.fixture
def base_valid_payload():
    """Payload dasar yang PASTI valid untuk semua field wajib ApplicantModeNew."""
    return {
        "name_contract_type": "Cash loans",
        "code_gender": "M",
        "flag_own_car": "N",
        "flag_own_realty": "Y",
        "name_type_suite": "Unaccompanied",
        "name_income_type": "Working",
        "name_education_type": "Higher education",
        "name_family_status": "Single / not married",
        "name_housing_type": "House / apartment",
        "status_pekerjaan": "Bekerja",
        "occupation_type": "Core staff",
        "organization_type": "Business Entity Type 3",
        "tanggal_mulai_kerja": date(2021, 1, 1),
        "cnt_children": 0,
        "cnt_fam_members": 1,
        "amt_income_total": 3_000_000,
        "amt_credit": 5_000_000,
        "amt_annuity": 500_000,   # tenor = 10 bulan, valid
        "tanggal_lahir": date(1995, 1, 1),  # ~31 tahun, aman untuk semua aturan usia
    }


class TestIncomeValidation:
    def test_income_below_minimum_should_fail(self, base_valid_payload):
        base_valid_payload["amt_income_total"] = 1_499_999
        with pytest.raises(ValidationError):
            ApplicantModeNew(**base_valid_payload)

    def test_income_at_minimum_should_pass(self, base_valid_payload):
        base_valid_payload["amt_income_total"] = 1_500_000
        assert ApplicantModeNew(**base_valid_payload)


class TestCreditValidation:
    def test_credit_below_minimum_should_fail(self, base_valid_payload):
        base_valid_payload["amt_credit"] = 999_999
        with pytest.raises(ValidationError):
            ApplicantModeNew(**base_valid_payload)

    def test_credit_at_minimum_should_pass(self, base_valid_payload):
        base_valid_payload["amt_credit"] = 1_000_000
        base_valid_payload["amt_annuity"] = 100_000  # tenor = 10 bulan, valid
        assert ApplicantModeNew(**base_valid_payload)


class TestTenorImplisitValidation:
    @pytest.mark.parametrize("credit, annuity, should_fail", [
        (10_000_000, 10_000_000, False),   # tenor = 1 bulan, batas bawah, valid
        (10_000_000, 10_000_001, True),    # tenor < 1 bulan, harus gagal
        (12_000_000, 100_000, False),      # tenor = 120 bulan PERSIS, batas atas, valid
        (12_000_000, 99_999, True),        # tenor = 120,0012 bulan, harus gagal
    ])
    def test_tenor_boundaries(self, base_valid_payload, credit, annuity, should_fail):
        base_valid_payload["amt_credit"] = credit
        base_valid_payload["amt_annuity"] = annuity
        if should_fail:
            with pytest.raises(ValidationError):
                ApplicantModeNew(**base_valid_payload)
        else:
            assert ApplicantModeNew(**base_valid_payload)


class TestFamilyCompositionValidation:
    def test_fam_members_less_than_children_should_fail(self, base_valid_payload):
        base_valid_payload["cnt_children"] = 2
        base_valid_payload["cnt_fam_members"] = 2  # minimal harus 3 (diri + 2 anak)
        with pytest.raises(ValidationError):
            ApplicantModeNew(**base_valid_payload)

    def test_fam_members_meets_minimum_should_pass(self, base_valid_payload):
        base_valid_payload["cnt_children"] = 2
        base_valid_payload["cnt_fam_members"] = 3
        assert ApplicantModeNew(**base_valid_payload)


class TestAgeValidation:
    def test_age_below_general_minimum_should_fail(self, base_valid_payload):
        # Status Single, usia dibuat ~18 tahun (di bawah batas umum 21 tahun)
        base_valid_payload["name_family_status"] = "Single / not married"
        base_valid_payload["tanggal_lahir"] = date(date.today().year - 18, 1, 1)
        base_valid_payload["status_pekerjaan"] = "Tidak Bekerja"
        base_valid_payload["occupation_type"] = None
        base_valid_payload["organization_type"] = None
        base_valid_payload["tanggal_mulai_kerja"] = None
        with pytest.raises(ValidationError):
            ApplicantModeNew(**base_valid_payload)

    def test_age_at_general_minimum_should_pass(self, base_valid_payload):
        base_valid_payload["name_family_status"] = "Single / not married"
        base_valid_payload["tanggal_lahir"] = date(date.today().year - 25, 1, 1)
        assert ApplicantModeNew(**base_valid_payload)


class TestEmploymentSyncValidation:
    def test_bekerja_without_occupation_should_fail(self, base_valid_payload):
        base_valid_payload["status_pekerjaan"] = "Bekerja"
        base_valid_payload["occupation_type"] = None
        with pytest.raises(ValidationError):
            ApplicantModeNew(**base_valid_payload)

    def test_tidak_bekerja_auto_fills_should_pass(self, base_valid_payload):
        base_valid_payload["status_pekerjaan"] = "Tidak Bekerja"
        base_valid_payload["occupation_type"] = None
        base_valid_payload["organization_type"] = None
        base_valid_payload["tanggal_mulai_kerja"] = None
        result = ApplicantModeNew(**base_valid_payload)
        assert result.occupation_type == "Unknown_Occupation"
        assert result.organization_type == "Not_Working"