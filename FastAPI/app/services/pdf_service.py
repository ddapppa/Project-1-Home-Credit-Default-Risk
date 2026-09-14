"""
Modul pembuatan dokumen PDF hasil evaluasi risiko kredit.

PENTING (Opsi A yang disepakati): PDF dibuat dari data yang DIHITUNG ULANG
oleh backend (bukan data yang dikirim balik oleh client), demi keotentikan
dokumen — mencegah manipulasi hasil lewat DevTools/client-side tampering.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

MODEL_NAME = "CatBoost v3"
MODEL_VERSION = "3.0"

BRAND_COLOR = colors.HexColor("#0A2E5C")
APPROVE_COLOR = colors.HexColor("#059669")
REVIEW_COLOR = colors.HexColor("#D97706")
DISCLAIMER_BG = colors.HexColor("#FFF7E6")
DISCLAIMER_BORDER = colors.HexColor("#F97316")


def _generate_reference_number(mode: str, identifier: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"BK-EVAL-{mode.upper()}-{identifier}-{timestamp}"


def build_prediction_pdf(mode: str, result: dict, sk_id_curr: int | None = None) -> bytes:
    """
    Bangun PDF dari hasil prediksi.

    mode: "existing" atau "new"
    result: dict berisi keputusan, probability_default, threshold,
            disclaimer, reasons (opsional), context (opsional)
    sk_id_curr: hanya untuk Mode Existing
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=20 * mm, bottomMargin=20 * mm,
        leftMargin=20 * mm, rightMargin=20 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BrandTitle", fontSize=20, leading=24, textColor=BRAND_COLOR, fontName="Helvetica-Bold", spaceAfter=2))
    styles.add(ParagraphStyle(name="SubTitle", fontSize=11, leading=15, textColor=colors.HexColor("#475569")))
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=12, textColor=BRAND_COLOR, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="BodyText2", fontSize=10, textColor=colors.HexColor("#334155"), leading=14))
    styles.add(ParagraphStyle(name="DecisionApprove", fontSize=22, textColor=APPROVE_COLOR, fontName="Helvetica-Bold", alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="DecisionReview", fontSize=22, textColor=REVIEW_COLOR, fontName="Helvetica-Bold", alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="DisclaimerText", fontSize=9.5, textColor=colors.HexColor("#7C2D12"), leading=13))
    styles.add(ParagraphStyle(name="Footer", fontSize=8, textColor=colors.HexColor("#94A3B8"), leading=11))

    elements = []

    identifier = str(sk_id_curr) if sk_id_curr else "BARU"
    ref_number = _generate_reference_number(mode, identifier)
    now_str = datetime.now().strftime("%d %B %Y, %H:%M WIB")

    # ---- Header ----
    elements.append(Paragraph("BankKu", styles["BrandTitle"]))
    elements.append(Spacer(1, 2 * mm))
    elements.append(Paragraph("Laporan Hasil Evaluasi Risiko Kredit", styles["SubTitle"]))
    elements.append(Spacer(1, 4 * mm))
    elements.append(HRFlowable(width="100%", color=BRAND_COLOR, thickness=1.5))
    elements.append(Spacer(1, 4 * mm))

    # ---- Metadata ----
    meta_rows = [
        ["Nomor Referensi Dokumen", ref_number],
        ["Tanggal & Waktu Cetak", now_str],
        ["Mode Evaluasi", "Nasabah Existing" if mode == "existing" else "Nasabah Baru"],
    ]
    if sk_id_curr:
        meta_rows.append(["SK_ID_CURR", str(sk_id_curr)])
    meta_rows.append(["Model yang Digunakan", f"{MODEL_NAME} (versi {MODEL_VERSION})"])

    meta_table = Table(meta_rows, colWidths=[65 * mm, 100 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#64748B")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#1A1A1A")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 8 * mm))

    # ---- Disclaimer (Mode Baru / OOD) ----
    disclaimer = result.get("disclaimer")
    if disclaimer:
        disc_table = Table(
            [[Paragraph(f"<b>PERINGATAN SISTEM:</b> {disclaimer}", styles["DisclaimerText"])]],
            colWidths=[165 * mm],
        )
        disc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), DISCLAIMER_BG),
            ("BOX", (0, 0), (-1, -1), 1.5, DISCLAIMER_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        elements.append(disc_table)
        elements.append(Spacer(1, 8 * mm))

    # ---- Keputusan ----
    is_approve = result.get("keputusan") == "APPROVE"
    decision_text = "DISETUJUI" if is_approve else "PERLU VERIFIKASI MANUAL"
    decision_style = styles["DecisionApprove"] if is_approve else styles["DecisionReview"]
    elements.append(Paragraph(decision_text, decision_style))
    elements.append(Spacer(1, 6 * mm))

    proba = result.get("probability_default", 0) * 100
    threshold = result.get("threshold", 0.6669) * 100
    prob_table = Table([
        ["Probabilitas Risiko Gagal Bayar", f"{proba:.2f}%"],
        ["Ambang Batas Keputusan (Threshold)", f"{threshold:.2f}%"],
    ], colWidths=[95 * mm, 70 * mm])
    prob_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    elements.append(prob_table)
    elements.append(Spacer(1, 8 * mm))

    # ---- Reasons (Mode Existing saja) ----
    reasons = result.get("reasons") or []
    if reasons:
        decrease = [r for r in reasons if r.get("direction") == "menurunkan_risiko"]
        increase = [r for r in reasons if r.get("direction") == "menaikkan_risiko"]

        if decrease:
            elements.append(Paragraph("Faktor yang Menurunkan Risiko", styles["SectionHeader"]))
            for r in decrease:
                elements.append(Paragraph(f"• {r['label']}: {r['value_display']}", styles["BodyText2"]))

        if increase:
            elements.append(Paragraph("Faktor yang Menaikkan Risiko", styles["SectionHeader"]))
            for r in increase:
                elements.append(Paragraph(f"• {r['label']}: {r['value_display']}", styles["BodyText2"]))

        elements.append(Spacer(1, 4 * mm))

    # ---- Context (Mode Baru) ----
    context = result.get("context")
    if context and (context.get("tujuan_dana") or context.get("rencana_pembayaran")):
        elements.append(Paragraph("Informasi Konteks Pengajuan (Non-Teknis)", styles["SectionHeader"]))
        elements.append(Paragraph(f"<b>Tujuan Dana:</b> {context.get('tujuan_dana') or '-'}", styles["BodyText2"]))
        elements.append(Paragraph(f"<b>Rencana Pembayaran:</b> {context.get('rencana_pembayaran') or '-'}", styles["BodyText2"]))
        elements.append(Spacer(1, 4 * mm))

    elements.append(Spacer(1, 10 * mm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#E2E8F0"), thickness=1))
    elements.append(Spacer(1, 6 * mm))

    # ---- Kolom Tanda Tangan ----
    sig_table = Table([
        ["_______________________", "_______________________"],
        ["Nama & Tanda Tangan Reviewer", "Nama & Tanda Tangan Approver"],
        ["Tanggal: ______________", "Tanggal: ______________"],
    ], colWidths=[82 * mm, 82 * mm])
    sig_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#64748B")),
        ("TOPPADDING", (0, 0), (-1, 0), 20),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(sig_table)
    elements.append(Spacer(1, 8 * mm))

    # ---- Footer ----
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#E2E8F0"), thickness=1))
    elements.append(Spacer(1, 3 * mm))
    elements.append(Paragraph(
        "Dokumen ini digenerate otomatis oleh sistem machine learning BankKu Credit Risk System dan "
        "TIDAK menggantikan keputusan resmi yang memerlukan otorisasi pejabat berwenang. "
        "Probabilitas yang ditampilkan adalah estimasi statistik, bukan kepastian mutlak.",
        styles["Footer"],
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()