const API_BASE = "";

const FIELD_LABELS = {
  name_contract_type: "Jenis Kontrak",
  code_gender: "Jenis Kelamin",
  flag_own_car: "Status Kepemilikan Mobil",
  own_car_age: "Usia Mobil",
  flag_own_realty: "Status Kepemilikan Rumah/Tanah",
  name_type_suite: "Pendamping Saat Mengajukan",
  tanggal_lahir: "Tanggal Lahir",
  cnt_children: "Jumlah Anak",
  cnt_fam_members: "Jumlah Anggota Keluarga",
  status_pekerjaan: "Status Pekerjaan",
  occupation_type: "Jenis Pekerjaan",
  organization_type: "Jenis Organisasi/Perusahaan",
  tanggal_mulai_kerja: "Tanggal Mulai Kerja",
  name_income_type: "Jenis Penghasilan",
  name_education_type: "Pendidikan Terakhir",
  name_family_status: "Status Pernikahan",
  name_housing_type: "Tipe Hunian",
  amt_income_total: "Total Penghasilan per Bulan",
  amt_credit: "Jumlah Kredit Diajukan",
  amt_annuity: "Angsuran per Periode",
  amt_goods_price: "Harga Barang",
  sk_id_curr: "SK ID Curr",
  applicant: "Data pengajuan",
};

const EVER_MARRIED_STATUSES = ["Married", "Civil marriage", "Separated", "Widow"];
const MIN_AGE_GENERAL = 21;
const MIN_AGE_MARRIED = 19;
const MAX_AGE = 100;
const TENOR_MIN_BULAN = 1;
const TENOR_MAX_BULAN = 120;

let lastSubmittedPayload = null;
let lastSubmittedEndpoint = null;

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    hideResult();
    hideError();

    const subtext = document.getElementById("tab-subtext");
    if (subtext) {
      subtext.textContent = btn.dataset.tab === "existing"
        ? "Evaluasi instan berbasis data riwayat kredit dan portofolio internal nasabah terdaftar."
        : "Penilaian kelayakan kredit komprehensif melalui formulir multi-tahap dan verifikasi parameter.";
    }
  });
});

document.getElementById("flag_own_car").addEventListener("change", (e) => {
  document.getElementById("field-own-car-age").classList.toggle("hidden", e.target.value !== "Y");
});
document.getElementById("sk_id_curr").addEventListener("input", hideError);

function syncIncomeTypeOptions(isWorking, isUserTriggered = true) {
  const select = document.getElementById("name_income_type");
  const options = select.querySelectorAll("option[data-employment]");
  let currentStillValid = false;

  options.forEach(opt => {
    const matches = isWorking
      ? opt.dataset.employment === "working"
      : opt.dataset.employment === "not-working";
    opt.hidden = !matches;
    opt.disabled = !matches;
    if (matches && opt.value === select.value) currentStillValid = true;
  });

  if (currentStillValid) {
    clearFieldError(select);
    return;
  }

  if (isWorking) {
    select.value = "Working";
    clearFieldError(select);
  } else {
    select.value = "";
    if (isUserTriggered) {
      showFieldError(select, "Jenis Penghasilan wajib dipilih ulang (misalnya Pensiunan) karena Status Pekerjaan berubah.");
    }
  }
}

document.getElementById("status_pekerjaan").addEventListener("change", (e) => {
  const isWorking = e.target.value === "Bekerja";
  document.getElementById("fields-bekerja").classList.toggle("hidden", !isWorking);

  const occ = document.querySelector('[name="occupation_type"]');
  const org = document.querySelector('[name="organization_type"]');
  const tgl = document.querySelector('[name="tanggal_mulai_kerja"]');

  occ.required = isWorking;
  org.required = isWorking;
  tgl.required = isWorking;

  if (!isWorking) {
    tgl.value = "";
    occ.value = "";
    org.value = "";
    clearFieldError(tgl);
    clearFieldError(occ);
    clearFieldError(org);
  } else {
    validateTanggalMulaiKerja();
  }

  syncIncomeTypeOptions(isWorking);
});

function getFieldByName(name) {
  return document.querySelector(`[name="${name}"]`);
}

function fieldLabel(el) {
  return FIELD_LABELS[el.name] || el.closest(".field")?.querySelector("label")?.textContent.replace("*", "").trim() || el.name;
}

function showFieldError(el, msg) {
  let err = el.parentElement.querySelector(".field-error");
  if (!err) {
    err = document.createElement("small");
    err.className = "field-error";
    el.insertAdjacentElement("afterend", err);
  }
  err.textContent = msg;
  el.classList.add("input-error");
}

function clearFieldError(el) {
  const err = el.parentElement.querySelector(".field-error");
  if (err) err.remove();
  el.classList.remove("input-error");
}

function checkNumberField(el) {
  if (el.closest(".hidden")) { clearFieldError(el); return true; }

  if (el.required && el.value === "") {
    showFieldError(el, `${fieldLabel(el)} wajib diisi.`);
    return false;
  }
  if (el.value === "") { clearFieldError(el); return true; }

  const val = Number(el.value);
  if (el.min !== "" && val < Number(el.min)) {
    showFieldError(el, `${fieldLabel(el)} tidak boleh kurang dari ${el.min}. Minimal ${el.min}.`);
    return false;
  }
  if (el.max !== "" && val > Number(el.max)) {
    showFieldError(el, `${fieldLabel(el)} tidak boleh lebih dari ${el.max}.`);
    return false;
  }

  clearFieldError(el);
  return true;
}

function checkGenericRequiredField(el) {
  if (el.closest(".hidden")) { clearFieldError(el); return true; }

  if (el.required && el.value === "") {
    showFieldError(el, `${fieldLabel(el)} wajib diisi.`);
    return false;
  }
  clearFieldError(el);
  return true;
}

function validateTanggalLahir() {
  const el = getFieldByName("tanggal_lahir");
  if (el.closest(".hidden")) { clearFieldError(el); return true; }

  if (!el.value) {
    if (el.required) { showFieldError(el, `${fieldLabel(el)} wajib diisi.`); return false; }
    clearFieldError(el);
    return true;
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const lahir = new Date(el.value);

  if (lahir > today) {
    showFieldError(el, "Tanggal Lahir tidak boleh di masa depan.");
    return false;
  }

  const usiaTahun = (today - lahir) / (1000 * 60 * 60 * 24 * 365.25);

  if (usiaTahun > MAX_AGE) {
    showFieldError(el, `Usia tidak wajar (${usiaTahun.toFixed(1)} tahun). Maksimal ${MAX_AGE} tahun.`);
    return false;
  }

  const famStatus = getFieldByName("name_family_status")?.value;
  const everMarried = EVER_MARRIED_STATUSES.includes(famStatus);
  const minAge = everMarried ? MIN_AGE_MARRIED : MIN_AGE_GENERAL;

  if (usiaTahun < minAge) {
    if (everMarried) {
      showFieldError(el, `Usia (${usiaTahun.toFixed(1)} tahun) di bawah batas minimal ${minAge} tahun untuk status "${famStatus}".`);
    } else {
      showFieldError(el, `Usia (${usiaTahun.toFixed(1)} tahun) di bawah batas minimal ${minAge} tahun untuk pemohon belum menikah.`);
    }
    return false;
  }

  clearFieldError(el);
  return true;
}

function validateFamilyComposition() {
  const childrenEl = getFieldByName("cnt_children");
  const famEl = getFieldByName("cnt_fam_members");
  if (childrenEl.value === "" || famEl.value === "") return true;

  const children = Number(childrenEl.value);
  const fam = Number(famEl.value);
  const minimalFam = children + 1;

  if (fam < minimalFam) {
    showFieldError(
      famEl,
      `Jumlah Anggota Keluarga (${fam}) tidak konsisten dengan Jumlah Anak (${children}). Minimal ${minimalFam} (diri sendiri + anak yang ditanggung).`
    );
    return false;
  }
  clearFieldError(famEl);
  return true;
}

getFieldByName("cnt_children").addEventListener("input", validateFamilyComposition);
getFieldByName("cnt_fam_members").addEventListener("input", validateFamilyComposition);

function validateTenorImplisit() {
    const creditEl = document.querySelector('[name="amt_credit"]');
    const annuityEl = document.querySelector('[name="amt_annuity"]');
    const credit = Number(creditEl.value);
    const annuity = Number(annuityEl.value);

    if (!credit || !annuity) return true; // biarkan checkNumberField menangani kosong/invalid

    const tenorBulan = credit / annuity;

    if (tenorBulan < TENOR_MIN_BULAN) {
        showFieldError(annuityEl,
            `Angsuran terlalu besar dibanding kredit (tenor implisit ${tenorBulan.toFixed(1)} bulan). ` +
            `Minimal tenor ${TENOR_MIN_BULAN} bulan.`);
        return false;
    }
    if (tenorBulan > TENOR_MAX_BULAN) {
        showFieldError(annuityEl,
            `Angsuran terlalu kecil dibanding kredit (tenor implisit ${tenorBulan.toFixed(0)} bulan). ` +
            `Maksimal tenor ${TENOR_MAX_BULAN} bulan (10 tahun).`);
        return false;
    }
    clearFieldError(annuityEl);
    return true;
}

document.querySelector('[name="amt_credit"]').addEventListener('input', validateTenorImplisit);
document.querySelector('[name="amt_annuity"]').addEventListener('input', validateTenorImplisit);

function validateTanggalMulaiKerja() {
  const el = getFieldByName("tanggal_mulai_kerja");
  if (el.closest(".hidden")) { clearFieldError(el); return true; }

  if (!el.value) {
    if (el.required) { showFieldError(el, `${fieldLabel(el)} wajib diisi.`); return false; }
    clearFieldError(el);
    return true;
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const mulai = new Date(el.value);

  if (mulai > today) {
    showFieldError(el, "Tanggal Mulai Kerja tidak boleh di masa depan.");
    return false;
  }

  const lahirEl = getFieldByName("tanggal_lahir");
  if (lahirEl.value) {
    const lahir = new Date(lahirEl.value);
    const usiaMinKerja = new Date(lahir);
    usiaMinKerja.setFullYear(usiaMinKerja.getFullYear() + 15);
    if (mulai < usiaMinKerja) {
      showFieldError(el, "Tanggal Mulai Kerja tidak wajar — usia nasabah belum 15 tahun saat itu.");
      return false;
    }
  }

  clearFieldError(el);
  return true;
}

document.querySelectorAll('#form-new input[type="number"]').forEach(el => {
  el.addEventListener("input", () => checkNumberField(el));
  el.addEventListener("blur", () => checkNumberField(el));
});

document.querySelectorAll('#form-new select').forEach(el => {
  el.addEventListener("change", () => checkGenericRequiredField(el));
});

getFieldByName("tanggal_lahir").addEventListener("change", () => {
  validateTanggalLahir();
  validateTanggalMulaiKerja();
});
getFieldByName("tanggal_mulai_kerja").addEventListener("change", validateTanggalMulaiKerja);
getFieldByName("name_family_status").addEventListener("change", validateTanggalLahir);

syncIncomeTypeOptions(document.getElementById("status_pekerjaan").value === "Bekerja", false);

function validateStep(stepNum) {
  const stepEl = document.querySelector(`.form-step[data-step="${stepNum}"]`);
  const fields = stepEl.querySelectorAll("input, select");
  let firstInvalid = null;

  for (const el of fields) {
    if (el.disabled || el.closest(".hidden")) { clearFieldError(el); continue; }

    let ok = true;
    if (el.name === "tanggal_lahir") {
      ok = validateTanggalLahir();
    } else if (el.name === "tanggal_mulai_kerja") {
      ok = validateTanggalMulaiKerja();
    } else if (el.name === "cnt_fam_members") {
      ok = checkNumberField(el) && validateFamilyComposition();
    } else if (el.name === "amt_annuity") {
      ok = checkNumberField(el) && validateTenorImplisit();
    } else if (el.type === "number") {
      ok = checkNumberField(el);
    } else {
      ok = checkGenericRequiredField(el);
    }

    if (!ok && !firstInvalid) firstInvalid = el;
  }

  if (firstInvalid) {
    firstInvalid.focus();
    showError("Periksa kembali kolom yang bertanda merah sebelum melanjutkan.");
    return false;
  }

  hideError();
  return true;
}

document.querySelectorAll(".btn-next").forEach(btn => {
  btn.addEventListener("click", () => {
    const currentStep = parseInt(btn.closest(".form-step").dataset.step);
    if (!validateStep(currentStep)) return;
    goToStep(parseInt(btn.dataset.next));
  });
});
document.querySelectorAll(".btn-prev").forEach(btn => {
  btn.addEventListener("click", () => goToStep(parseInt(btn.dataset.prev)));
});

function goToStep(stepNum) {
  document.querySelectorAll(".form-step").forEach(s => s.classList.remove("active"));
  document.querySelector(`.form-step[data-step="${stepNum}"]`).classList.add("active");

  document.querySelectorAll(".step").forEach(s => {
    const n = parseInt(s.dataset.step);
    s.classList.remove("active", "done");
    if (n < stepNum) s.classList.add("done");
    if (n === stepNum) s.classList.add("active");
  });

  if (stepNum === 4) renderReviewSummary();
}

function renderReviewSummary() {
  const form = document.getElementById("form-new");
  const data = new FormData(form);
  let html = "<ul>";
  for (const [key, val] of data.entries()) {
    if (val) html += `<li><strong>${key}</strong>: ${val}</li>`;
  }
  html += "</ul>";
  document.getElementById("review-summary").innerHTML = html;
}

document.getElementById("btn-check-existing").addEventListener("click", async () => {
  const skInput = document.getElementById("sk_id_curr");
  if (!checkNumberField(skInput)) return;
  if (!skInput.value) return showError("SK ID Curr wajib diisi.");
  await submitPrediction("/predict/existing", { sk_id_curr: parseInt(skInput.value) }, "existing");
});

document.getElementById("form-new").addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!validateStep(1) || !validateStep(2) || !validateStep(3)) return;

  const form = e.target;
  const data = new FormData(form);
  const applicant = {};
  const context = {};

  const contextFields = ["tujuan_dana", "rencana_pembayaran"];

  for (const [key, val] of data.entries()) {
    if (contextFields.includes(key)) {
      context[key] = val || null;
    } else if (val !== "") {
      if (["cnt_children", "cnt_fam_members", "own_car_age"].includes(key)) {
        applicant[key] = parseInt(val);
      } else if (["amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price"].includes(key)) {
        applicant[key] = parseFloat(val);
      } else {
        applicant[key] = val;
      }
    }
  }
  context.catatan_reviewer = null;

  await submitPrediction("/predict/new", { applicant, context }, "new");
});

function friendlyValidationMessage(loc, msg) {
  const field = loc?.[loc.length - 1];
  const label = FIELD_LABELS[field] || field || "Data";
  const lower = (msg || "").toLowerCase();

  if (lower.startsWith("value error,")) {
    return msg.replace(/^value error,\s*/i, "");
  }
  if (lower.includes("greater than or equal to")) {
    const min = msg.match(/[\d.]+$/)?.[0];
    return `${label} tidak boleh kurang dari ${min}.`;
  }
  if (lower.includes("less than or equal to")) {
    const max = msg.match(/[\d.]+$/)?.[0];
    return `${label} tidak boleh lebih dari ${max}.`;
  }
  if (lower.includes("field required") || lower.includes("missing")) {
    return `${label} wajib diisi.`;
  }
  if (lower.includes("valid integer") || lower.includes("valid number")) {
    return `${label} harus berupa angka yang valid.`;
  }
  return `${label}: ${msg}`;
}

async function submitPrediction(endpoint, body, mode) {
  lastSubmittedPayload = body;
  lastSubmittedEndpoint = endpoint;

  hideError();
  showLoading(true);
  const submitBtns = document.querySelectorAll("#btn-check-existing, #btn-submit-new");
  submitBtns.forEach(b => b.disabled = true);

  try {
    const res = await fetch(API_BASE + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    let json = null;
    try {
      json = await res.json();
    } catch (_parseErr) {
      json = null;
    }

    if (!res.ok) {
      if (res.status === 404) {
        showError(json?.detail || "Data tidak ditemukan.");
      } else if (res.status === 422) {
        const messages = (json?.detail || []).map(d => friendlyValidationMessage(d.loc, d.msg));
        showError(messages.length ? messages.join(" ") : "Data yang dikirim tidak valid.");
      } else if (res.status >= 500) {
        showError("Server mengalami kendala internal. Ini kemungkinan bug backend, bukan kesalahan input Anda — coba lagi atau laporkan ke tim teknis.");
      } else {
        showError("Terjadi kesalahan tak terduga. Coba lagi.");
      }
      return;
    }

    if (!json) {
      showError("Respons server tidak valid. Coba lagi.");
      return;
    }

    renderResult(json, mode);
  } catch (err) {
    showError("Gagal terhubung ke server. Periksa apakah server backend sedang berjalan.");
  } finally {
    showLoading(false);
    submitBtns.forEach(b => b.disabled = false);
  }
}

function renderResult(data, mode) {
  const resultSection = document.getElementById("result");
  const statusCard = document.getElementById("status-card");
  const banner = document.getElementById("disclaimer-banner");

  resultSection.classList.remove("hidden");
  resultSection.scrollIntoView({ behavior: "smooth" });

  if (data.disclaimer) {
    banner.textContent = data.disclaimer;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }

  const isApprove = data.keputusan === "APPROVE";
  statusCard.className = "status-card " + (isApprove ? "approve" : "review");
  document.getElementById("status-title").textContent = isApprove ? "Disetujui" : "Perlu Verifikasi Manual";
  document.getElementById("status-detail").textContent = isApprove
    ? "Profil nasabah berada di bawah ambang batas risiko yang ditetapkan."
    : "Skor risiko atau kondisi data memerlukan tinjauan manual sebelum keputusan final.";

  const proba = data.probability_default;
  const threshold = data.threshold;
  document.getElementById("gauge-fill").style.width = `${(1 - proba) * 100}%`;
  document.getElementById("gauge-threshold").style.left = `${threshold * 100}%`;
  document.getElementById("gauge-label").textContent =
    `Probabilitas Risiko Gagal Bayar: ${(proba * 100).toFixed(1)}% (ambang batas: ${(threshold * 100).toFixed(2)}%)`;

  const ctxBox = document.getElementById("context-summary");
  if (data.context && (data.context.tujuan_dana || data.context.rencana_pembayaran)) {
    ctxBox.classList.remove("hidden");
    ctxBox.innerHTML = `
      <p><strong>Tujuan Dana:</strong> ${data.context.tujuan_dana || "-"}</p>
      <p><strong>Rencana Pembayaran:</strong> ${data.context.rencana_pembayaran || "-"}</p>
    `;
  } else {
    ctxBox.classList.add("hidden");
  }

  const reasonsBox = document.getElementById("reasons-summary");
  if (data.reasons && data.reasons.length > 0) {
    const decreaseList = document.getElementById("reasons-decrease");
    const increaseList = document.getElementById("reasons-increase");
    decreaseList.innerHTML = "";
    increaseList.innerHTML = "";

    const decreaseItems = data.reasons.filter(r => r.direction === "menurunkan_risiko");
    const increaseItems = data.reasons.filter(r => r.direction === "menaikkan_risiko");

    decreaseItems.forEach(r => {
      const li = document.createElement("li");
      li.textContent = `${r.label}: ${r.value_display}`;
      decreaseList.appendChild(li);
    });
    increaseItems.forEach(r => {
      const li = document.createElement("li");
      li.textContent = `${r.label}: ${r.value_display}`;
      increaseList.appendChild(li);
    });

    document.getElementById("reasons-decrease-group").classList.toggle("hidden", decreaseItems.length === 0);
    document.getElementById("reasons-increase-group").classList.toggle("hidden", increaseItems.length === 0);
    reasonsBox.classList.remove("hidden");
  } else {
    reasonsBox.classList.add("hidden");
  }

  const resetBtn = document.getElementById("btn-reset");
  if (mode === "existing") {
    resetBtn.textContent = "Cek ID Lain";
    resetBtn.onclick = () => {
      hideResult();
      const skInput = document.getElementById("sk_id_curr");
      skInput.value = "";
      skInput.focus();
    };
  } else {
    resetBtn.textContent = "Ajukan Baru";
    resetBtn.onclick = () => {
      hideResult();
      document.getElementById("form-new").reset();
      document.querySelectorAll(".field-error").forEach(e => e.remove());
      document.querySelectorAll(".input-error").forEach(e => e.classList.remove("input-error"));
      goToStep(1);
    };
  }
}

async function downloadPdf() {
  if (!lastSubmittedPayload || !lastSubmittedEndpoint) {
    showError("Belum ada hasil prediksi untuk dijadikan PDF. Cek skor risiko dulu.");
    return;
  }

  const reportEndpoint = lastSubmittedEndpoint + "/report";
  const btn = document.getElementById("btn-download-pdf");
  btn.disabled = true;

  try {
    const res = await fetch(API_BASE + reportEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastSubmittedPayload),
    });

    if (!res.ok) {
      showError("Gagal membuat PDF (status " + res.status + "). Coba lagi.");
      return;
    }

    const blob = await res.blob();
    console.log("PDF diterima:", blob.size, "bytes, tipe:", blob.type);
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "laporan-evaluasi-risiko-kredit.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();

    showPdfFallbackLink(url);
  } catch (err) {
    console.error(err);
    showError("Gagal terhubung ke server saat membuat PDF.");
  } finally {
    btn.disabled = false;
  }
}

function showPdfFallbackLink(url) {
  let link = document.getElementById("pdf-fallback-link");
  if (!link) {
    link = document.createElement("a");
    link.id = "pdf-fallback-link";
    link.className = "block text-center text-sm text-sky-700 underline mt-3";
    document.getElementById("btn-download-pdf").insertAdjacentElement("afterend", link);
  }
  link.href = url;
  link.download = "laporan-evaluasi-risiko-kredit.pdf";
  link.textContent = "PDF siap. Kalau unduhan tidak otomatis mulai, klik di sini.";
  link.classList.remove("hidden");
}

document.getElementById("btn-download-pdf").addEventListener("click", downloadPdf);

if (window.lucide) {
  lucide.createIcons();
}

function hideResult() { document.getElementById("result").classList.add("hidden"); }
function showLoading(show) { document.getElementById("loading").classList.toggle("hidden", !show); }
function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = msg;
  box.classList.remove("hidden");
}
function hideError() { document.getElementById("error-box").classList.add("hidden"); }