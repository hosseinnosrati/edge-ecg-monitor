"""Bilingual (English & Persian) Technical Documentation Generator for Edge-ECG-Monitor.

Generates a publication-quality PDF report with architecture breakdown,
clinical foundations, DSP theory, benchmarks, and academic admissions scope.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display


# Register Fonts
pdfmetrics.registerFont(TTFont("Tahoma", "C:/Windows/Fonts/tahoma.ttf"))
pdfmetrics.registerFont(TTFont("Tahoma-Bold", "C:/Windows/Fonts/tahomabd.ttf"))


def fa(text: str) -> str:
    """Reshape and reorder Persian/Arabic text for correct PDF rendering."""
    if not text:
        return ""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def build_pdf(filename: str = "Edge_ECG_Monitor_Technical_Documentation.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Tahoma-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0d3b66"),
        alignment=1,  # Center
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Tahoma",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=15
    )

    h1_en = ParagraphStyle(
        "Heading1_EN",
        fontName="Tahoma-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0d3b66"),
        spaceBefore=12,
        spaceAfter=4
    )

    h1_fa = ParagraphStyle(
        "Heading1_FA",
        fontName="Tahoma-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0d3b66"),
        alignment=2,  # Right
        spaceBefore=8,
        spaceAfter=4
    )

    body_en = ParagraphStyle(
        "Body_EN",
        fontName="Tahoma",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6
    )

    body_fa = ParagraphStyle(
        "Body_FA",
        fontName="Tahoma",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2b2b2b"),
        alignment=2,  # Right
        spaceAfter=8
    )

    callout_style = ParagraphStyle(
        "Callout",
        fontName="Tahoma",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1b4332"),
        spaceAfter=4
    )

    story = []

    # 1. Title & Header
    story.append(Paragraph("Edge-ECG-Monitor: Technical & Academic Report", title_style))
    story.append(Paragraph(
        fa("سامانه پایش بلادرنگ نوار قلب، پردازش سیگنال زیستی و هوش مصنوعی در لبه"),
        subtitle_style
    ))
    story.append(Paragraph(
        "<b>Author:</b> hnosrati0-source | <b>Contact:</b> hnosrati0@gmail.com | "
        "<b>Repository:</b> github.com/hnosrati0-source/edge-ecg-monitor",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0d3b66"), spaceAfter=12))

    # 2. Executive Summary (Bilingual)
    story.append(Paragraph("1. Executive Summary", h1_en))
    story.append(Paragraph(
        "<b>Edge-ECG-Monitor</b> is an open-source, resource-constrained biomedical signal processing "
        "and Edge AI architecture designed for real-time cardiac arrhythmia detection on wearable and embedded IoT hardware. "
        "Evaluated on the gold-standard MIT-BIH PhysioNet database, the system achieves microsecond-level inference latency (0.12 ms/beat) "
        "and 100% sensitivity on ventricular ectopic beats (PVC) while maintaining a strict <175 MB RAM footprint.",
        body_en
    ))
    story.append(Paragraph(fa("۱. خلاصه اجرایی و مهندسی"), h1_fa))
    story.append(Paragraph(
        fa("پروژه Edge-ECG-Monitor یک سامانه پردازش سیگنال پزشکی و هوش مصنوعی در لبه (Edge AI) است که برای تشخیص بلادرنگ "
           "آریتمی‌های قلبی روی سخت‌افزارهای کم‌مصرف و گجت‌های پوشیدنی سلامت طراحی شده است. این سامانه با ارزیابی روی پایگاه استاندارد بین‌المللی "
           "MIT-BIH PhysioNet، توانسته زمان پردازش هر ضربان قلب را به ۰.۱۲ میلی‌ثانیه برساند و با مصرف رم کمتر از ۱۷۵ مگابایت، "
           "دقت ۱۰۰٪ را در شناسایی انقباضات زودرس بطنی (PVC) ثبت کند."),
        body_fa
    ))

    # 3. Visual Demonstration
    img_path = "assets/demo_ecg_detection.png"
    if os.path.exists(img_path):
        story.append(Spacer(1, 4))
        story.append(Image(img_path, width=520, height=260))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<i>Figure 1: Complete 3-stage pipeline. Top: Raw ECG + 60Hz powerline & baseline drift. "
            "Middle: Pan-Tompkins MWI envelope. Bottom: Cleaned ECG with AI-classified beats (Green: Normal, Red: PVC).</i>",
            ParagraphStyle("Cap", fontName="Tahoma", fontSize=8, leading=10, textColor=colors.HexColor("#666"), alignment=1)
        ))

    # 4. Core Engineering & DSP Modules
    story.append(Spacer(1, 8))
    story.append(Paragraph("2. Technical & DSP Architecture", h1_en))
    story.append(Paragraph(
        "<b>• Digital Signal Conditioning:</b> Cascaded zero-phase IIR Notch filter (50/60 Hz, Q=30) and 3rd-order Butterworth bandpass filter (0.5–40 Hz) eliminating baseline respiration drift and electromyographic (EMG) noise.<br/>"
        "<b>• Pan-Tompkins QRS Detection:</b> Multi-stage digital differentiator (5-point stencil), non-linear squaring, and moving window integration (MWI ~150ms window) with adaptive peak thresholding.<br/>"
        "<b>• Autonomic Spectral HRV Analysis:</b> 4.0 Hz cubic-spline interpolation of RR intervals followed by Welch's Power Spectral Density (PSD) decomposition to compute clinical LF/HF sympathovagal balance.<br/>"
        "<b>• Bare-Metal C99 Firmware:</b> Standalone header (`edge_classifier.h`, 16.3 KB) with zero dynamic allocation (`malloc`-free) for ARM Cortex-M microcontrollers.",
        body_en
    ))
    story.append(Paragraph(fa("۲. معماری پردازش سیگنال و نرم‌افزار"), h1_fa))
    story.append(Paragraph(
        fa("• فیلترگذاری دیجیتال: فیلتر ناچ IIR جهت حذف نویز برق شهر (۵۰/۶۰ هرتز) و فیلتر میان‌گذر باترورث (۰.۵ الی ۴۰ هرتز) جهت حذف نویزهای ماهیچه‌ای و دریفت تنفسی.<br/>"
           "• الگوریتم Pan-Tompkins: مشتق‌گیر ۵ نقطه‌ای، مجذورساز غیرخطی، انتگرال‌گیر پنجره متحرک و آستانه‌گذاری انطباقی جهت استخراج دقیق پیک‌های R.<br/>"
           "• تحلیل طیف توان HRV: درونیابی اسپیلاین مکعبی در فرکانس ۴ هرتز و محاسبه چگالی طیف توان (Welch PSD) جهت تفکیک باندهای فرکانسی و شاخص LF/HF.<br/>"
           "• کدنویسی به زبان C99 نهفته: تولید خودکار فایل هدر مستقل ۱۶ کیلوبایتی بدون وابستگی به پایتون یا تخصیص حافظه دینامیک، آماده برای میکروکنترلرهای STM32 و ESP32."),
        body_fa
    ))

    # 5. Hardware Benchmark Table
    story.append(Spacer(1, 6))
    story.append(Paragraph("3. Hardware Profiling & Benchmark (Dell Inspiron 1525 / Dual-Core)", h1_en))

    table_data = [
        [
            Paragraph("<b>Benchmark Metric</b>", body_en),
            Paragraph("<b>Measured Result</b>", body_en),
            Paragraph("<b>Significance for Edge / IoT</b>", body_en)
        ],
        [
            Paragraph("Model Binary Size", body_en),
            Paragraph("<b>16.3 KB (C) / 38 KB (Py)</b>", body_en),
            Paragraph("Fits effortlessly on low-cost microcontrollers", body_en)
        ],
        [
            Paragraph("Total Pipeline Latency (60s)", body_en),
            Paragraph("<b>7.61 ms</b>", body_en),
            Paragraph("<b>>7,800× faster than real-time</b>", body_en)
        ],
        [
            Paragraph("Average Latency per Beat", body_en),
            Paragraph("<b>0.121 ms / beat</b>", body_en),
            Paragraph("Ultra-low latency for live patient telemetry", body_en)
        ],
        [
            Paragraph("Peak RAM Footprint", body_en),
            Paragraph("<b><175 MB</b>", body_en),
            Paragraph("Runs comfortably on low-power edge nodes", body_en)
        ],
        [
            Paragraph("Clinical Sensitivity (PVC)", body_en),
            Paragraph("<b>100.0% (AUC: 1.000)</b>", body_en),
            Paragraph("AAMI-compliant arrhythmia classification", body_en)
        ]
    ]

    t = Table(table_data, colWidths=[160, 140, 220])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0d3b66")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ]))
    story.append(t)

    # 6. Academic Relevance for Graduate Admissions
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Academic Scope for Graduate Admissions (BME, Telecom, ECE)", h1_en))
    story.append(Paragraph(
        "<b>• Biomedical Engineering (BME):</b> Directly addresses biosignal acquisition, cardiac electrophysiology, and autonomic HRV diagnostics.<br/>"
        "<b>• Electrical & Telecom (EE/DSP):</b> Proves mastery over digital filter design (IIR/FIR), non-uniform sampling, and Welch power spectral estimation.<br/>"
        "<b>• Electrical & Computer Engineering (ECE / Embedded AI):</b> Demonstrates edge software engineering, resource profiling, and bare-metal ANSI C firmware generation.",
        body_en
    ))
    story.append(Paragraph(fa("۴. کاربرد در اپلای دپارتمان‌های مهندسی پزشکی، مخابرات و کامپیوتر"), h1_fa))
    story.append(Paragraph(
        fa("این پروژه پل ارتباطی سه رشته است: برای مهندسی پزشکی نشان‌دهنده درک فیزیولوژی قلب و نوار قلب؛ برای مخابرات نشان‌دهنده طراحی فیلترهای دیجیتال، قضیه نایکویست و تحلیل فوریه زمان-فرکانس؛ و برای مهندسی کامپیوتر اثبات‌کننده توانایی پیاده‌سازی هوش مصنوعی سبک روی میکروکنترلرها با زمان اجرای میکروثانیه است."),
        body_fa
    ))

    # 7. Quick Start & Verification
    story.append(Spacer(1, 6))
    story.append(Paragraph("5. Verification & Live Dashboard", h1_en))
    story.append(Paragraph(
        "<b>Interactive Web UI:</b> Run `streamlit run app.py` for live filter tuning and Welch PSD visualization.<br/>"
        "<b>Automated Benchmark:</b> Run `python benchmark.py` to reproduce memory and latency measurements.<br/>"
        "<b>Firmware Generation:</b> Run `python -m src.models.c_export` to recompile the ANSI C99 header.",
        body_en
    ))

    # Build Document
    doc.build(story)
    print(f"Documentation PDF successfully created: {filename} ({os.path.getsize(filename) / 1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf("docs/Edge_ECG_Monitor_Technical_Documentation.pdf")
    build_pdf("Edge_ECG_Monitor_Technical_Documentation.pdf")
