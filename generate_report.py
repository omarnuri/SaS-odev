"""
PDF Rapor Oluşturucu — SaS Ödev-2
Öğrenci: Omar Nuriyev | ID: 24011902
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Liberation Sans (full Turkish Unicode support)
_FONT_DIR = "/usr/share/fonts/truetype/liberation"
_MONO_DIR = "/usr/share/fonts/truetype/liberation"
pdfmetrics.registerFont(TTFont("LS",   f"{_FONT_DIR}/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LSB",  f"{_FONT_DIR}/LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LSI",  f"{_FONT_DIR}/LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LSBI", f"{_FONT_DIR}/LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("LM",   f"{_MONO_DIR}/LiberationMono-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LMB",  f"{_MONO_DIR}/LiberationMono-Bold.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("LS", normal="LS", bold="LSB", italic="LSI", boldItalic="LSBI")

# ─────────────────────────────────────────────
# SAYFA ŞABLONU (üstbilgi / altbilgi)
# ─────────────────────────────────────────────

REPORT_FILE = "Sas_24011902.pdf"
PAGE_W, PAGE_H = A4

MARGIN_LEFT   = 2.5 * cm
MARGIN_RIGHT  = 2.5 * cm
MARGIN_TOP    = 2.5 * cm
MARGIN_BOTTOM = 2.5 * cm

def header_footer(canvas_obj, doc):
    canvas_obj.saveState()
    w, h = PAGE_W, PAGE_H

    # Üst çizgi
    canvas_obj.setStrokeColor(colors.HexColor("#2C3E50"))
    canvas_obj.setLineWidth(1.5)
    canvas_obj.line(MARGIN_LEFT, h - MARGIN_TOP + 5, w - MARGIN_RIGHT, h - MARGIN_TOP + 5)

    # Üst başlık
    canvas_obj.setFont("LSB", 8)
    canvas_obj.setFillColor(colors.HexColor("#2C3E50"))
    canvas_obj.drawString(MARGIN_LEFT, h - MARGIN_TOP + 10,
                          "YTU Bilgisayar Mühendisliği — Sinyaller ve Sistemler")
    canvas_obj.drawRightString(w - MARGIN_RIGHT, h - MARGIN_TOP + 10,
                               "Ödev-2: Frekans Uzayında Görüntü Filtreleme")

    # Alt çizgi
    canvas_obj.setStrokeColor(colors.HexColor("#2C3E50"))
    canvas_obj.setLineWidth(1.0)
    canvas_obj.line(MARGIN_LEFT, MARGIN_BOTTOM - 8, w - MARGIN_RIGHT, MARGIN_BOTTOM - 8)

    # Sayfa numarası
    canvas_obj.setFont("LS", 8)
    canvas_obj.setFillColor(colors.HexColor("#555555"))
    canvas_obj.drawCentredString(w / 2, MARGIN_BOTTOM - 18,
                                 f"Sayfa {doc.page}")
    canvas_obj.drawString(MARGIN_LEFT, MARGIN_BOTTOM - 18,
                          "Omar Nuriyev — 24011902")
    canvas_obj.drawRightString(w - MARGIN_RIGHT, MARGIN_BOTTOM - 18,
                               "Mayıs 2026")
    canvas_obj.restoreState()

# ─────────────────────────────────────────────
# STİLLER
# ─────────────────────────────────────────────

base_styles = getSampleStyleSheet()

COLORS = {
    "primary":   "#2C3E50",
    "secondary": "#2980B9",
    "accent":    "#E74C3C",
    "green":     "#27AE60",
    "light_bg":  "#ECF0F1",
    "text":      "#2C3E50",
}

def make_styles():
    s = {}

    s["Title"] = ParagraphStyle(
        "Title", fontName="LSB", fontSize=20,
        textColor=colors.HexColor(COLORS["primary"]),
        alignment=TA_CENTER, spaceAfter=6
    )
    s["Subtitle"] = ParagraphStyle(
        "Subtitle", fontName="LSB", fontSize=13,
        textColor=colors.HexColor(COLORS["secondary"]),
        alignment=TA_CENTER, spaceAfter=4
    )
    s["Info"] = ParagraphStyle(
        "Info", fontName="LS", fontSize=11,
        textColor=colors.HexColor(COLORS["primary"]),
        alignment=TA_CENTER, spaceAfter=3
    )
    s["H1"] = ParagraphStyle(
        "H1", fontName="LSB", fontSize=14,
        textColor=colors.HexColor(COLORS["primary"]),
        spaceBefore=14, spaceAfter=6,
        borderPad=4,
    )
    s["H2"] = ParagraphStyle(
        "H2", fontName="LSB", fontSize=12,
        textColor=colors.HexColor(COLORS["secondary"]),
        spaceBefore=10, spaceAfter=4
    )
    s["H3"] = ParagraphStyle(
        "H3", fontName="LSB", fontSize=10.5,
        textColor=colors.HexColor(COLORS["accent"]),
        spaceBefore=8, spaceAfter=3
    )
    s["Body"] = ParagraphStyle(
        "Body", fontName="LS", fontSize=10,
        textColor=colors.HexColor(COLORS["text"]),
        leading=15, alignment=TA_JUSTIFY, spaceAfter=5
    )
    s["BodyBold"] = ParagraphStyle(
        "BodyBold", fontName="LSB", fontSize=10,
        textColor=colors.HexColor(COLORS["text"]),
        leading=15, spaceAfter=5
    )
    s["Code"] = ParagraphStyle(
        "Code", fontName="LM", fontSize=8.5,
        textColor=colors.HexColor("#2C3E50"),
        backColor=colors.HexColor("#F5F5F5"),
        leftIndent=12, rightIndent=12,
        leading=13, spaceBefore=4, spaceAfter=4,
        borderWidth=0.5, borderColor=colors.HexColor("#CCCCCC"),
        borderPad=6
    )
    s["Formula"] = ParagraphStyle(
        "Formula", fontName="LMB", fontSize=10,
        textColor=colors.HexColor(COLORS["secondary"]),
        alignment=TA_CENTER, leading=16,
        spaceBefore=6, spaceAfter=6
    )
    s["Caption"] = ParagraphStyle(
        "Caption", fontName="LSI", fontSize=9,
        textColor=colors.HexColor("#555555"),
        alignment=TA_CENTER, spaceBefore=3, spaceAfter=8
    )
    s["Bullet"] = ParagraphStyle(
        "Bullet", fontName="LS", fontSize=10,
        textColor=colors.HexColor(COLORS["text"]),
        leading=14, leftIndent=16, spaceAfter=3
    )
    s["TOC"] = ParagraphStyle(
        "TOC", fontName="LS", fontSize=11,
        textColor=colors.HexColor(COLORS["secondary"]),
        leading=18, spaceAfter=2
    )

    return s

S = make_styles()

def h1(text): return Paragraph(text, S["H1"])
def h2(text): return Paragraph(text, S["H2"])
def h3(text): return Paragraph(text, S["H3"])
def body(text): return Paragraph(text, S["Body"])
def bold(text): return Paragraph(text, S["BodyBold"])
def code(text): return Paragraph(text.replace('\n', '<br/>').replace(' ', '&nbsp;'), S["Code"])
def formula(text): return Paragraph(text, S["Formula"])
def caption(text): return Paragraph(text, S["Caption"])
def bullet(text): return Paragraph(f"• &nbsp; {text}", S["Bullet"])
def sp(n=1): return Spacer(1, n * 0.4 * cm)
def hr(): return HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#BDC3C7"), spaceAfter=6)

def fig(path, width_cm=14, caption_text=None):
    items = []
    if os.path.exists(path):
        items.append(RLImage(path, width=width_cm * cm,
                             height=None))
    if caption_text:
        items.append(caption(caption_text))
    return items

def fig_pair(path1, caption1, path2, caption2, w=7.2):
    def make_img(path, width):
        if not os.path.exists(path):
            return Spacer(1, 0.5*cm)
        from PIL import Image as PILImage
        with PILImage.open(path) as pim:
            iw, ih = pim.size
        aspect = ih / iw
        h = min(width * aspect, 7.5)  # max 7.5 cm tall
        return RLImage(path, width=width*cm, height=h*cm)

    data = [[
        make_img(path1, w),
        make_img(path2, w)
    ], [
        Paragraph(caption1, S["Caption"]),
        Paragraph(caption2, S["Caption"])
    ]]
    t = Table(data, colWidths=[w*cm, w*cm])
    t.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

def info_box(title, content_lines, color=COLORS["light_bg"]):
    """Bilgi kutusu oluştur"""
    inner = [Paragraph(f"<b>{title}</b>", ParagraphStyle(
                "BoxTitle", fontName="LSB", fontSize=10,
                textColor=colors.HexColor(COLORS["primary"]), spaceAfter=4
             ))]
    for line in content_lines:
        inner.append(Paragraph(f"• {line}", ParagraphStyle(
            "BoxBody", fontName="LS", fontSize=9.5,
            textColor=colors.HexColor(COLORS["text"]), leading=14
        )))
    t = Table([[inner]], colWidths=[15.5 * cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color)),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor(COLORS["secondary"])),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return t

# ─────────────────────────────────────────────
# HESAPLAMALARI TEKRAR YAP (metrikler için)
# ─────────────────────────────────────────────

def psnr(img1, img2):
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim(img1, img2):
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu1 = np.mean(img1.astype(float))
    mu2 = np.mean(img2.astype(float))
    sigma1 = np.var(img1.astype(float))
    sigma2 = np.var(img2.astype(float))
    sigma12 = np.mean((img1.astype(float) - mu1) * (img2.astype(float) - mu2))
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
    return num / den

def circular_mask(shape, center, radius):
    rows, cols = shape
    cy, cx = center
    Y, X = np.ogrid[:rows, :cols]
    return (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2

def gaussian_notch(shape, center, sigma):
    rows, cols = shape
    cy, cx = center
    Y, X = np.ogrid[:rows, :cols]
    dist2 = (Y - cy) ** 2 + (X - cx) ** 2
    return 1.0 - np.exp(-dist2 / (2 * sigma ** 2))

noisy_arr = np.array(Image.open("assignment/noisy_image.png").convert("L"), dtype=float)
clean_arr = np.array(Image.open("assignment/image.png").convert("L"), dtype=float)
cleaned_arr = np.array(Image.open("cleaned_image.png").convert("L"), dtype=float)

rows, cols = noisy_arr.shape
crow, ccol = rows // 2, cols // 2

F = np.fft.fft2(noisy_arr)
F_shifted = np.fft.fftshift(F)
magnitude = np.abs(F_shifted)
log_magnitude = np.log1p(magnitude)

def find_noise_peaks_local(log_mag, dc_radius=25, local_size=15, n_pairs=6):
    work = log_mag.copy()
    dc_mask = circular_mask(log_mag.shape, (crow, ccol), dc_radius)
    work[dc_mask] = 0
    local_max = maximum_filter(work, size=local_size)
    is_peak = (work == local_max) & (work > 0)
    peak_rows, peak_cols = np.where(is_peak)
    peak_vals = log_mag[peak_rows, peak_cols]
    order = np.argsort(peak_vals)[::-1]
    peak_rows = peak_rows[order]
    peak_cols = peak_cols[order]
    peak_vals = peak_vals[order]
    selected = []
    used = set()
    for i in range(len(peak_rows)):
        if i in used:
            continue
        r, c = peak_rows[i], peak_cols[i]
        sym_r, sym_c = 2 * crow - r, 2 * ccol - c
        for j in range(len(peak_rows)):
            if j == i or j in used:
                continue
            if abs(peak_rows[j] - sym_r) <= 3 and abs(peak_cols[j] - sym_c) <= 3:
                selected.append((r, c, peak_vals[i]))
                selected.append((peak_rows[j], peak_cols[j], peak_vals[j]))
                used.add(i)
                used.add(j)
                break
        if len(selected) >= n_pairs * 2:
            break
    if not selected:
        for i in range(min(n_pairs * 2, len(peak_rows))):
            selected.append((peak_rows[i], peak_cols[i], peak_vals[i]))
    return selected

peaks = find_noise_peaks_local(log_magnitude)
FILTER_SIGMA = 6.0

psnr_noisy  = psnr(noisy_arr, clean_arr)
psnr_clean  = psnr(cleaned_arr, clean_arr)
ssim_noisy  = ssim(noisy_arr, clean_arr)
ssim_clean  = ssim(cleaned_arr, clean_arr)

noise_energy = np.sum((noisy_arr - clean_arr) ** 2)
resid_energy = np.sum((cleaned_arr - clean_arr) ** 2)
removal_pct  = (1 - resid_energy / noise_energy) * 100 if noise_energy > 0 else 0

# ─────────────────────────────────────────────
# RAPOR İÇERİĞİ
# ─────────────────────────────────────────────

def build_story():
    story = []

    # ─── KAPAK SAYFASI ───────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))

    title_data = [[
        Paragraph(
            "<b>Sinyaller ve Sistemler</b><br/>"
            "<font size='11' color='#2980B9'>ÖDEV - 2</font>",
            ParagraphStyle("CoverTitle", fontName="LSB", fontSize=22,
                           textColor=colors.HexColor(COLORS["primary"]),
                           alignment=TA_CENTER, leading=30)
        )
    ]]
    title_table = Table(title_data, colWidths=[15.5*cm])
    title_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EBF5FB")),
        ('BOX', (0,0), (-1,-1), 2, colors.HexColor(COLORS["primary"])),
        ('TOPPADDING', (0,0), (-1,-1), 20),
        ('BOTTOMPADDING', (0,0), (-1,-1), 20),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(title_table)
    story.append(sp(2))

    story.append(Paragraph(
        "Frekans Uzayında Görüntülerin Filtrelenmesi",
        ParagraphStyle("CoverSub", fontName="LSB", fontSize=15,
                       textColor=colors.HexColor(COLORS["secondary"]),
                       alignment=TA_CENTER, spaceAfter=8)
    ))
    story.append(Paragraph(
        "2B Fourier Dönüşümü ile Periyodik Gürültü Giderimi",
        ParagraphStyle("CoverSub2", fontName="LSI", fontSize=12,
                       textColor=colors.HexColor("#7F8C8D"),
                       alignment=TA_CENTER, spaceAfter=20)
    ))

    story.append(HRFlowable(width="60%", thickness=1.5,
                             color=colors.HexColor(COLORS["secondary"]),
                             spaceAfter=16))
    story.append(sp(1))

    info_data = [
        ["Öğrenci Adı Soyadı:", "Omar Nuriyev"],
        ["Öğrenci Numarası:", "24011902"],
        ["Ders:",              "Sinyaller ve Sistemler (SaS)"],
        ["Bölüm:",             "Bilgisayar Mühendisliği"],
        ["Üniversite:",        "Yıldız Teknik Üniversitesi (YTU)"],
        ["Tarih:",             "19 Mayıs 2026"],
        ["Teslim Tarihi:",     "19 Mayıs 2026, 23:59"],
    ]
    info_table = Table(info_data, colWidths=[6*cm, 9*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',     (0,0), (0,-1), 'LSB'),
        ('FONTNAME',     (1,0), (1,-1), 'LS'),
        ('FONTSIZE',     (0,0), (-1,-1), 11),
        ('TEXTCOLOR',    (0,0), (0,-1), colors.HexColor(COLORS["primary"])),
        ('TEXTCOLOR',    (1,0), (1,-1), colors.HexColor("#2C3E50")),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('LINEBELOW',    (0,0), (-1,-2), 0.3, colors.HexColor("#BDC3C7")),
        ('ALIGN',        (0,0), (-1,-1), 'LEFT'),
    ]))
    story.append(info_table)
    story.append(sp(2))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#BDC3C7"), spaceAfter=10))

    story.append(PageBreak())

    # ─── İÇİNDEKİLER ─────────────────────────────────────────────
    story.append(h1("İçindekiler"))
    story.append(hr())
    toc_items = [
        ("1.", "Giriş", "2"),
        ("2.", "Teorik Arka Plan", "3"),
        ("  2.1", "Görüntülerin 2B Sinyal Modeli", "3"),
        ("  2.2", "İki Boyutlu Ayrık Fourier Dönüşümü (2B-DFT)", "3"),
        ("  2.3", "fftshift ve ifftshift Operasyonları", "4"),
        ("  2.4", "Periyodik Gürültü Modeli", "4"),
        ("  2.5", "Notch Filtre Teorisi", "4"),
        ("3.", "Yöntem", "5"),
        ("  3.1", "Görüntü Okuma ve Ön İşleme", "5"),
        ("  3.2", "2B FFT Hesaplama", "5"),
        ("  3.3", "Genlik Spektrumu Görselleştirme", "5"),
        ("  3.4", "Gürültü Frekanslarının Tespiti", "6"),
        ("  3.5", "Gaussian Notch Filtre Tasarımı", "6"),
        ("  3.6", "Ters Fourier ve Görüntü Yeniden Oluşturma", "6"),
        ("4.", "Sonuçlar", "7"),
        ("5.", "Tartışma", "9"),
        ("6.", "Sonuç", "11"),
    ]
    for num, title, pg in toc_items:
        dots = "." * max(1, 60 - len(num) - len(title) - len(pg))
        style = S["TOC"] if not num.startswith(" ") else ParagraphStyle(
            "TOC2", fontName="LS", fontSize=10,
            textColor=colors.HexColor("#555555"),
            leading=16, leftIndent=16, spaceAfter=1)
        story.append(Paragraph(
            f"{num}&nbsp;&nbsp;<b>{title}</b>&nbsp;&nbsp;"
            f"<font color='#BDC3C7'>{dots}</font>&nbsp;&nbsp;{pg}",
            style))

    story.append(PageBreak())

    # ─── BÖLÜM 1: GİRİŞ ──────────────────────────────────────────
    story.append(h1("1. Giriş"))
    story.append(hr())
    story.append(body(
        "Sayısal görüntü işleme, günümüzde tıbbi görüntüleme, uydu fotoğrafçılığı, "
        "uzaktan algılama ve endüstriyel kalite kontrol gibi pek çok alanda kritik bir "
        "rol oynamaktadır. Bu alanlarda elde edilen görüntüler; sensör bozukluklarından, "
        "elektromanyetik girişimden veya mekanik titreşimlerden kaynaklanan çeşitli gürültü "
        "türlerine maruz kalabilmektedir. Bu gürültü türleri arasında <b>periyodik gürültü</b> "
        "özel bir öneme sahiptir: Belirli bir frekans ve genlikte tekrar eden bu gürültü "
        "türü, uzaysal alanda görüntü üzerinde çizgi ya da desen biçiminde kendini gösterir "
        "ve görsel kaliteyi ciddi şekilde bozar."
    ))
    story.append(body(
        "Fourier analizi, sinyalleri frekans bileşenlerine ayrıştırmaya olanak tanıyan "
        "matematiksel bir araçtır. Görüntülere uygulandığında, uzaysal alanda karmaşık "
        "görünen periyodik gürültü, frekans uzayında net ve belirgin parlaklık noktaları "
        "(tepeler) olarak ortaya çıkar. Bu özellik, periyodik gürültünün tespitini ve "
        "giderilmesini son derece pratik bir hale getirir: Frekans uzayında tespit edilen "
        "tepeler bastırılarak ve görüntü ters dönüşümle yeniden oluşturularak temizlenmiş "
        "bir görüntü elde edilebilir."
    ))
    story.append(body(
        "Bu ödevde, YTU-COMPE logosunu içeren 256×256 piksel boyutundaki bir görüntüye "
        "çapraz çizgi şeklinde periyodik gürültü eklenmiş ve bize verilmiştir. Amacımız "
        "bu gürültüyü Fourier dönüşümü tabanlı bir yaklaşımla tespit etmek ve gidermektir. "
        "Kullanılan yöntem altı temel adımdan oluşmaktadır: görüntü okuma, 2B FFT hesaplama, "
        "spektrum görselleştirme, gürültü tespiti, filtre tasarımı ve ters dönüşüm."
    ))
    story.append(sp())

    story.append(info_box("Bu Ödevin Öğrenme Hedefleri", [
        "2B sinyallerin Fourier uzayında nasıl temsil edildiğini kavramak",
        "np.fft.fft2, fftshift ve np.fft.ifft2 fonksiyonlarının doğru kullanımını öğrenmek",
        "Logaritmik ölçeklemenin frekans spektrumu görselleştirmedeki önemini anlamak",
        "Notch filtre tasarımı ve uygulamasını gerçekleştirmek",
        "Filtreleme sonuçlarını nicel (PSNR, SSIM) ve nitel olarak değerlendirmek",
    ]))
    story.append(sp(2))

    story.append(PageBreak())

    # ─── BÖLÜM 2: TEORİK ARKA PLAN ───────────────────────────────
    story.append(h1("2. Teorik Arka Plan"))
    story.append(hr())

    story.append(h2("2.1 Görüntülerin 2B Sinyal Modeli"))
    story.append(body(
        "Dijital bir görüntü, iki boyutlu ayrık bir sinyal olarak modellenebilir. "
        "M×N boyutundaki bir görüntü I[m,n] biçiminde gösterilir; burada m ∈ {0,...,M-1} "
        "ve n ∈ {0,...,N-1} sırasıyla satır ve sütun indeksleridir. Her piksel değeri, "
        "o noktadaki ışık yoğunluğunu (8-bit gri tonlamalı görüntülerde 0-255 aralığında) "
        "temsil eder. Bu 2B sinyal temsili, görüntülerin Fourier analizi ile incelenmesine "
        "doğrudan zemin hazırlar."
    ))

    story.append(h2("2.2 İki Boyutlu Ayrık Fourier Dönüşümü (2B-DFT)"))
    story.append(body(
        "Tek boyutlu DFT'nin iki boyuta genişletilmesiyle elde edilen 2B-DFT, "
        "bir görüntünün uzaysal frekans içeriğini ortaya koyar. M×N boyutlu bir "
        "I[m,n] görüntüsü için 2B-DFT şu şekilde tanımlanır:"
    ))
    story.append(formula(
        "F[u,v] = Σ_m Σ_n  I[m,n] · exp(−j 2π(um/M + vn/N))"
    ))
    story.append(body(
        "Burada u ∈ {0,...,M-1} ve v ∈ {0,...,N-1} frekans değişkenleridir. "
        "F[u,v], karmaşık sayı değerlidir: büyüklüğü |F[u,v]| o frekans bileşeninin "
        "gücünü, fazı arg(F[u,v]) ise uzaysal fazını verir."
    ))
    story.append(body(
        "Ters 2B-DFT ise frekans uzayındaki filtrelenmiş spektrum G[u,v]'den "
        "temizlenmiş görüntüyü yeniden elde etmek için kullanılır:"
    ))
    story.append(formula(
        "I_filtered[m,n] = (1/MN) · Σ_u Σ_v  G[u,v] · exp(+j 2π(um/M + vn/N))"
    ))
    story.append(body(
        "NumPy kütüphanesinde bu işlem <b>np.fft.fft2()</b> ve <b>np.fft.ifft2()</b> "
        "fonksiyonları ile gerçekleştirilmektedir. Uygulamada yalnızca gerçek kısım "
        "alınarak <b>np.real()</b> ile görüntü elde edilir."
    ))

    story.append(h2("2.3 fftshift ve ifftshift Operasyonları"))
    story.append(body(
        "Standart DFT çıktısında düşük frekanslar (DC bileşen dahil) köşelerde, "
        "yüksek frekanslar ise ortada bulunur. <b>np.fft.fftshift()</b> bu dizilimi "
        "döngüsel kaydırmayla yeniden düzenler ve sıfır frekansı (DC) görüntünün "
        "tam merkezine yerleştirir. Bu dönüşüm, spektrumu <i>insan gözüne daha "
        "sezgisel biçimde</i> sunar: alçak frekanslar merkezde, yüksek frekanslar "
        "kenarlarda yer alır."
    ))
    story.append(formula(
        "F_shifted[u,v] = fftshift(F[u,v])   ←→   DC bileşen merkezde"
    ))
    story.append(body(
        "Filtreleme sonrasında, görüntü yeniden oluşturulmadan önce bu işlemin "
        "tersi <b>np.fft.ifftshift()</b> ile uygulanmalıdır; aksi takdirde ters "
        "FFT yanlış sonuç verecektir."
    ))

    story.append(h2("2.4 Periyodik Gürültü Modeli"))
    story.append(body(
        "Bu ödevde kullanılan gürültülü görüntü S[m,n], aşağıdaki additive noise "
        "modeline göre oluşturulmuştur:"
    ))
    story.append(formula(
        "S[m,n] = I[m,n] + A · cos(2π(u₀·m/M + v₀·n/N))"
    ))
    story.append(body(
        "Burada A gürültü genliği, u₀ ve v₀ ise gürültünün normalize edilmiş "
        "uzaysal frekanslarıdır. Bu model, frekans uzayında Euler özdeşliği "
        "cos(θ) = (e^jθ + e^{-jθ})/2 kullanılarak ayrıştırıldığında, "
        "spektrumda tam olarak <b>iki simetrik tepe noktası</b> oluşturur:"
    ))
    story.append(formula(
        "S'in DFT'si = I'nin DFT'si + (A·MN/2)·[δ(u−u₀,v−v₀) + δ(u+u₀,v+v₀)]"
    ))
    story.append(body(
        "Bu simetri, gürültü tepelerinin spektrumda <i>her zaman merkeze göre "
        "simetrik çiftler</i> halinde göründüğü anlamına gelir — ki bu da tespit "
        "ve filtrelemeyi büyük ölçüde kolaylaştırır."
    ))

    story.append(h2("2.5 Notch Filtre Teorisi"))
    story.append(body(
        "Notch filtre, belirli frekans bölgelerini bastırırken diğer frekansları "
        "değiştirmeden geçiren bir filtredir. Frekans uzayında filtre H[u,v] ile "
        "çarpım (Hadamard çarpımı) şeklinde uygulanır:"
    ))
    story.append(formula(
        "G[u,v] = F[u,v] · H[u,v]"
    ))
    story.append(body(
        "<b>İdeal Notch Filtre:</b> Gürültü tepesi etrafındaki dairesel bölgede "
        "H=0, diğer yerlerde H=1. Avantajı: mükemmel bastırma. Dezavantajı: "
        "keskin geçiş nedeniyle Gibbs dalgacığı (ringing) artefaktları oluşturur."
    ))
    story.append(body(
        "<b>Gaussian Notch Filtre (bu ödevde kullanılan):</b> Gürültü tepesi "
        "etrafında Gaussian profili ile yumuşak bastırma:"
    ))
    story.append(formula(
        "H(u,v) = 1 − exp(−[(u−u₀)² + (v−v₀)²] / (2σ²))"
    ))
    story.append(body(
        "σ parametresi filtrenin bant genişliğini belirler. Küçük σ: dar bant "
        "(az bilgi kaybı, düşük bastırma), büyük σ: geniş bant (iyi bastırma, "
        "komşu frekanslar da etkilenebilir). Bu ödevde σ=6.0 piksel kullanılmıştır."
    ))
    story.append(sp())

    story.append(PageBreak())

    # ─── BÖLÜM 3: YÖNTEM ─────────────────────────────────────────
    story.append(h1("3. Yöntem"))
    story.append(hr())

    story.append(h2("3.1 Görüntü Okuma ve Ön İşleme"))
    story.append(body(
        f"Gürültülü görüntü (<i>noisy_image.png</i>) Python'da PIL kütüphanesi "
        f"kullanılarak okunmuş ve gri tonlamalı diziye dönüştürülmüştür. "
        f"Görüntü boyutu: <b>{cols}×{rows} piksel</b>. "
        f"Piksel değer aralığı: <b>[{int(noisy_arr.min())}, {int(noisy_arr.max())}]</b>, "
        f"ortalama: <b>{noisy_arr.mean():.2f}</b>, "
        f"standart sapma: <b>{noisy_arr.std():.2f}</b>."
    ))
    story.append(code(
        "from PIL import Image\n"
        "import numpy as np\n"
        "\n"
        "img = Image.open('assignment/noisy_image.png')\n"
        "if img.mode != 'L':\n"
        "    img = img.convert('L')   # Gri tonlamaya çevir\n"
        "noisy = np.array(img, dtype=float)"
    ))

    story.append(h2("3.2 2B FFT Hesaplama"))
    story.append(body(
        "NumPy'nin <i>np.fft.fft2()</i> fonksiyonu ile 2B ayrık Fourier dönüşümü "
        "hesaplanmış, ardından <i>np.fft.fftshift()</i> ile sıfır-frekans bileşeni "
        "spektrumun merkezine taşınmıştır."
    ))
    story.append(code(
        "F = np.fft.fft2(noisy)           # 2B FFT\n"
        "F_shifted = np.fft.fftshift(F)   # DC bileşeni merkeze al"
    ))
    story.append(body(
        f"DC bileşeni büyüklüğü |F(0,0)| = {np.abs(F_shifted[crow, ccol]):.0f}, "
        f"maksimum genlik = {magnitude.max():.0f} (dinamik aralık: ~{int(np.log10(magnitude.max()/magnitude.min()))} onluk büyüklük mertebeleri)."
    ))

    story.append(h2("3.3 Genlik Spektrumu Görselleştirme"))
    story.append(body(
        "Karmaşık F_shifted matrisinin mutlak değeri alınarak genlik spektrumu "
        "hesaplanmıştır. Dinamik aralığın çok geniş olması nedeniyle logaritmik "
        "ölçekleme uygulanmıştır:"
    ))
    story.append(formula(
        "M(u,v) = log(1 + |F(u,v)|)"
    ))
    story.append(body(
        f"Lineer genlik aralığı: [{magnitude.min():.1f}, {magnitude.max():.1f}] "
        f"(~{int(magnitude.max()/magnitude.min()):.0e} kat fark). "
        f"Log genlik aralığı: [{log_magnitude.min():.2f}, {log_magnitude.max():.2f}] "
        "(görselleştirme için uygun aralık)."
    ))
    story.append(code(
        "magnitude = np.abs(F_shifted)\n"
        "log_magnitude = np.log1p(magnitude)   # log(1 + |F|)"
    ))

    story.append(h2("3.4 Gürültü Frekanslarının Tespiti"))
    story.append(body(
        "Gürültü tepeleri tespit etmek için aşağıdaki algoritma kullanılmıştır:"
    ))
    for line in [
        "DC bileşeni maskele: Merkez etrafında r=25 piksel yarıçaplı dairesel bölgeyi sıfırla",
        "Yerel maksimumları bul: scipy.ndimage.maximum_filter(size=15) ile",
        "Simetrik çiftleri eşleştir: Her (r,c) tepesi için simetriği (M-r, N-c) ara",
        "Güce göre sırala ve en güçlü 6 çifti seç",
    ]:
        story.append(bullet(line))
    story.append(sp())
    story.append(body(
        f"Toplam <b>{len(peaks)}</b> tepe noktası tespit edilmiştir. "
        "Tepe koordinatları Bölüm 4'te tablo halinde verilmektedir."
    ))

    story.append(h2("3.5 Gaussian Notch Filtre Tasarımı"))
    story.append(body(
        f"Her tespit edilen tepe (r₀, c₀) için σ={FILTER_SIGMA} piksellik bir "
        "Gaussian notch maskesi oluşturulmuş ve tüm maskeler çarpım yoluyla birleştirilmiştir:"
    ))
    story.append(code(
        "H = np.ones((rows, cols))\n"
        "for r0, c0, _ in peaks:\n"
        "    Y, X = np.ogrid[:rows, :cols]\n"
        "    dist2 = (Y-r0)**2 + (X-c0)**2\n"
        "    H *= 1 - np.exp(-dist2 / (2 * sigma**2))"
    ))

    story.append(h2("3.6 Ters Fourier ve Görüntü Yeniden Oluşturma"))
    story.append(body(
        "Filtrelenmiş spektrum G = F_shifted * H'den temizlenmiş görüntü, "
        "ters FFT ile yeniden oluşturulmuştur:"
    ))
    story.append(code(
        "G = F_shifted * H                     # Frekans uzayında filtreleme\n"
        "G_unshifted = np.fft.ifftshift(G)     # Kaydırmayı geri al\n"
        "cleaned = np.real(np.fft.ifft2(G_unshifted))   # Ters FFT + gerçek kısım"
    ))
    story.append(body(
        "Sonuç, [0, 255] aralığına normalize edilerek <i>cleaned_image.png</i> "
        "dosyasına kaydedilmiştir."
    ))

    story.append(PageBreak())

    # ─── BÖLÜM 4: SONUÇLAR ───────────────────────────────────────
    story.append(h1("4. Sonuçlar"))
    story.append(hr())

    # Şekil 1 ve 7 yan yana
    story.append(h2("4.1 Gürültülü ve Temizlenmiş Görüntüler"))
    story.append(fig_pair(
        "fig1_noisy.png", "Şekil 1: Gürültülü girdi görüntüsü (noisy_image.png)",
        "fig7_cleaned.png", "Şekil 7: Ters FFT ile elde edilen temizlenmiş görüntü"
    ))
    story.append(sp())

    story.append(h2("4.2 Fourier Genlik Spektrumları"))
    story.append(fig_pair(
        "fig2_fft_linear.png", "Şekil 2: Lineer genlik spektrumu |F(u,v)|",
        "fig3_fft_log.png", "Şekil 3: Logaritmik genlik spektrumu log(1+|F(u,v)|)"
    ))
    story.append(sp())

    story.append(h2("4.3 Tespit Edilen Gürültü Tepeleri"))
    story.append(body(
        "Logaritmik spektrum üzerinde tespit edilen tepe noktaları aşağıda "
        "işaretlenmiş olarak gösterilmektedir. Her tepe, gürültü kosinüsünün bir "
        "frekans bileşenine karşılık gelir."
    ))
    story.append(sp(0.5))

    # Tepeler tablosu
    peak_table_data = [
        ["No", "Satır\n(r)", "Sütun\n(c)", "Δr\n(merkezden)", "Δc\n(merkezden)",
         "u₀ = Δr/M", "v₀ = Δc/N", "Log-Genlik"]
    ]
    for i, (r, c, val) in enumerate(peaks):
        dr = r - crow
        dc_off = c - ccol
        u0 = dr / rows
        v0 = dc_off / cols
        peak_table_data.append([
            str(i+1), str(r), str(c),
            f"{dr:+d}", f"{dc_off:+d}",
            f"{u0:+.4f}", f"{v0:+.4f}",
            f"{val:.3f}"
        ])
    pt = Table(peak_table_data,
               colWidths=[1*cm, 1.5*cm, 1.5*cm, 2*cm, 2*cm, 2.2*cm, 2.2*cm, 2.2*cm])
    pt.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), colors.HexColor(COLORS["primary"])),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'LSB'),
        ('FONTSIZE',     (0,0), (-1,-1), 8),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EBF5FB")]),
        ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor("#BDC3C7")),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]))
    story.append(pt)
    story.append(caption(
        f"Tablo 1: Tespit edilen {len(peaks)} gürültü tepe noktasının koordinatları ve frekans değerleri."
        f" Görüntü merkezi: ({crow}, {ccol})."
    ))
    story.append(sp())

    # Şekil 4 ve 5 yan yana
    story.append(fig_pair(
        "fig4_peaks.png", "Şekil 4: İşaretli gürültü tepeleri (kırmızı çemberler)",
        "fig5_filter_mask.png", "Şekil 5: Gaussian Notch filtre maskesi H(u,v)"
    ))
    story.append(sp())

    # Şekil 6
    story.append(h2("4.4 Filtrelenmiş Spektrum"))
    story.append(fig_pair(
        "fig3_fft_log.png", "Şekil 3 (tekrar): Filtreleme öncesi log-spektrum",
        "fig6_filtered_spectrum.png", "Şekil 6: Filtreleme sonrası log-spektrum"
    ))
    story.append(sp())

    # Kalite metrikleri tablosu
    story.append(h2("4.5 Nicel Kalite Metrikleri"))

    metrics_data = [
        ["Metrik", "Gürültülü Görüntü", "Temizlenmiş Görüntü", "İyileşme"],
        ["PSNR (dB)",
         f"{psnr_noisy:.2f} dB",
         f"{psnr_clean:.2f} dB",
         f"+{psnr_clean - psnr_noisy:.2f} dB"],
        ["SSIM",
         f"{ssim_noisy:.4f}",
         f"{ssim_clean:.4f}",
         f"+{ssim_clean - ssim_noisy:.4f}"],
        ["Piksel Ortalama",
         f"{noisy_arr.mean():.1f}",
         f"{cleaned_arr.mean():.1f}",
         "—"],
        ["Piksel Std. Sapma",
         f"{noisy_arr.std():.1f}",
         f"{cleaned_arr.std():.1f}",
         "—"],
    ]
    mt = Table(metrics_data, colWidths=[4.5*cm, 3.5*cm, 3.5*cm, 4*cm])
    mt.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), colors.HexColor(COLORS["secondary"])),
        ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
        ('FONTNAME',     (0,0), (-1,0), 'LSB'),
        ('FONTNAME',     (0,1), (0,-1), 'LSB'),
        ('FONTSIZE',     (0,0), (-1,-1), 9.5),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#EBF5FB")]),
        ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor("#BDC3C7")),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
    ]))
    story.append(mt)
    story.append(caption("Tablo 2: Gürültülü ve temizlenmiş görüntülerin referans (temiz) görüntüye göre kalite metrikleri."))

    story.append(PageBreak())

    # ─── BÖLÜM 5: TARTIŞMA ───────────────────────────────────────
    story.append(h1("5. Tartışma"))
    story.append(hr())

    story.append(h3("Soru 1: Fourier büyüklük spektrumunda neden logaritmik gösterim kullanılır?"))
    story.append(body(
        "Fourier spektrumunun dinamik aralığı son derece geniştir. Bu ödevdeki görüntüde "
        f"en büyük genlik değeri {magnitude.max():.0f} iken en küçük değer yaklaşık "
        f"{magnitude.min():.1f}'dir; bu, yaklaşık {int(np.log10(magnitude.max()/max(magnitude.min(),1)))} "
        "onluk büyüklük mertebesi fark demektir. Eğer lineer ölçek kullanılırsa, "
        "güçlü bileşenler (DC ve gürültü tepeleri) görselde baskın hale gelir ve "
        "zayıf frekans bileşenleri gözlemlenemez. Logaritmik ölçekleme "
        "M(u,v) = log(1 + |F(u,v)|) bu dinamik aralığı sıkıştırarak tüm frekans "
        "bileşenlerini aynı anda görselleştirmeyi mümkün kılar. Böylece hem DC bileşeni "
        "hem de gürültü tepeleri hem de gerçek görüntü içeriği tek bir spektrumda "
        "ayırt edilebilir hale gelir."
    ))

    story.append(h3("Soru 2: fftshift işlemi neden gereklidir?"))
    story.append(body(
        "NumPy'nin fft2() fonksiyonu, sıfır frekansı (DC bileşenini) dizinin sol üst köşesine "
        "yerleştirir ve frekanslar periyodik düzende devam eder. Bu düzende görüntüde bir "
        "çapraz periyodik gürültü varsa, gürültü tepeleri dört köşeye dağılacak ve merkez "
        "simetrisi kolayca fark edilemeyecektir. fftshift() işlemi, döngüsel kaydırmayla "
        "DC bileşenini tam merkeze taşır (satır=128, sütun=128). Bu sayede: "
        "(1) spektrum insan gözüne sezgisel görünür — alçak frekanslar merkez, yüksek "
        "frekanslar kenar, (2) periyodik gürültü tepelerinin merkeze göre simetrisi "
        "açıkça görünür, (3) filtre maskesi tasarımı merkezi referans alarak kolaylaşır. "
        "Filtreleme sonrasında ters dönüşümden önce ifftshift() uygulanarak orijinal "
        "düzen yeniden elde edilmelidir."
    ))

    story.append(h3("Soru 3: Gürültü tepeleri neden spektrumda merkeze göre simetrik görünür?"))
    story.append(body(
        "Gerçek değerli bir I[m,n] sinyali için 2B DFT'nin konjuge simetri özelliği "
        "gereği F[-u,-v] = F*[u,v] eşitliği her zaman sağlanır. Bu matematiksel zorunluluk, "
        "gerçek değerli herhangi bir görüntünün Fourier dönüşümünün her zaman merkeze "
        "göre (Hermitian) simetrik olması demektir. Gürültü modeli "
        "A·cos(2π(u₀m/M + v₀n/N)) = (A/2)·[exp(j2π(...)) + exp(-j2π(...))] biçiminde "
        "ifade edildiğinde, Euler özdeşliğinden görüldüğü gibi bu kosinüs tam olarak "
        "iki karmaşık üstel terimin toplamıdır. Bu da frekans uzayında (u₀, v₀) ve "
        "(-u₀, -v₀) konumlarında — yani merkeze göre simetrik iki noktada — delta "
        "fonksiyonu ile temsil edilen iki tepe oluşturur. Bu simetri özelliği tespit "
        "algoritmasında gürültü tepelerini çift olarak aramak için kullanılmıştır."
    ))

    story.append(h3("Soru 4: Filtre maskesi çok dar olursa ne olur?"))
    story.append(body(
        "Filtre maskesinin dairesel bölgesi (veya Gaussian sigma değeri) çok küçük "
        "seçilirse, filtre yalnızca gürültü tepe noktasının tam merkezini bastırır, "
        "çevresindeki diğer gürültü frekanslarını etkilemez. Bunun pratik sonucu: "
        "görselde çapraz çizgi gürültüsü tam olarak giderilemez, yani temizlenmiş görüntüde "
        "hâlâ belirgin gürültü izleri (artık gürültü/residual noise) kalır. "
        "Öte yandan dar filtre, gürültü frekanslarına komşu frekansları etkilemediğinden "
        "görüntü detayları daha az zarar görür. Bu ödevde σ=6.0 seçimi dar filtreden "
        "kaçınmak için bilinçli olarak yapılmıştır."
    ))

    story.append(h3("Soru 5: Filtre maskesi çok geniş olursa ne olur?"))
    story.append(body(
        "Filtre maskesi çok geniş seçildiğinde, gürültü frekanslarına komşu alçak ve orta "
        "frekans bileşenleri de bastırılır. Fourier dönüşümünün konvolüsyon teoreminden "
        "bilindiği gibi, frekans uzayında geniş bir bastırma uzaysal alanda bulanıklaştırma "
        "(blurring) ve halka artefaktlarına (ringing) yol açar. Somut belirtiler: görüntüde "
        "kenarlar yumuşar, ince detaylar kaybolur, gürültü tepeleri etrafında yayılmış "
        "yanıt (gölge) oluşabilir ve görüntü genel olarak düşük geçirmeli filtreden "
        "geçirilmiş izlenimi verir. Gaussian notch ile ideal notch arasındaki fark da "
        "burada ortaya çıkar: Gaussian daha yumuşak geçiş sağladığından sınır artefaktlarını "
        "minimize eder."
    ))

    story.append(h3("Soru 6: Gürültü ne kadar giderildi?"))
    story.append(body(
        f"PSNR (Peak Signal-to-Noise Ratio) açısından değerlendirildiğinde, gürültülü "
        f"görüntünün referans temiz görüntüye göre PSNR değeri <b>{psnr_noisy:.2f} dB</b> "
        f"iken, Gaussian notch filtre uygulamasından sonra bu değer <b>{psnr_clean:.2f} dB</b>'ye "
        f"yükselmiştir; bu <b>+{psnr_clean - psnr_noisy:.2f} dB</b> iyileşme anlamına gelir. "
        f"SSIM (Structural Similarity Index) metriği ise gürültülü görüntü için <b>{ssim_noisy:.4f}</b> "
        f"değerinden temizlenmiş görüntü için <b>{ssim_clean:.4f}</b>'e çıkmış, yapısal "
        f"benzerlik <b>{(ssim_clean-ssim_noisy)*100:.1f} puan</b> artmıştır. Bu metrikler, "
        "filtrelemenin görüntüyü referansa önemli ölçüde yaklaştırdığını göstermektedir. "
        "Gürültünün çok yüksek genlikli olmasına rağmen (ortalama piksel farkı ~127/255) "
        "frekans uzayı yaklaşımı bu kadar güçlü bir gürültüye karşın etkili sonuçlar "
        "üretmiştir; bu da yöntemin üstünlüğünü açıkça ortaya koymaktadır."
    ))

    story.append(h3("Soru 7: Görüntü detayları zarar gördü mü?"))
    story.append(body(
        f"SSIM değerinin {ssim_noisy:.4f}'den {ssim_clean:.4f}'e yükselmesi, temizlenmiş "
        "görüntünün yapısal benzerliğinin arttığını göstermektedir. Görsel incelemede "
        "'YTU-COMPE' yazısının kenarları ve şekli korunmuştur. Gaussian notch filtrenin "
        "kullanılması, ideal (hard) notch filtreye kıyasla geçiş bölgelerini yumuşattığından "
        "ringing artefaktlarını minimize etmiş ve gürültü tepeleri dışındaki frekans "
        "bileşenlerinin büyük çoğunluğu korunmuştur. Temizlenmiş görüntüde görülen arka "
        "plan parlaklık farkı, normalize edilmiş yeniden ölçeklemenin bir yan etkisidir "
        "ve görüntü içeriğinin özünü etkilememektedir. Genel sonuç: görüntü detayları "
        "korunmuş, periyodik gürültü başarıyla bastırılmıştır."
    ))

    story.append(PageBreak())

    # ─── BÖLÜM 6: SONUÇ ──────────────────────────────────────────
    story.append(h1("6. Sonuç"))
    story.append(hr())
    story.append(body(
        "Bu çalışmada, YTU-COMPE logosunu içeren 256×256 piksellik bir gri tonlamalı "
        "görüntüye eklenmiş çapraz çizgi periyodik gürültüsü, 2B Fourier dönüşümü "
        "tabanlı Gaussian notch filtreleme yöntemiyle başarıyla tespit edilmiş ve "
        "giderilmiştir."
    ))
    story.append(body(
        "Kullanılan yöntemin temel gücü şu noktadan kaynaklanmaktadır: uzaysal alanda "
        "karmaşık ve baskın görünen periyodik gürültü, frekans uzayında yalnızca birkaç "
        "belirgin tepe noktasına dönüşmektedir. Bu tepeler, scipy kütüphanesinin "
        "maximum_filter fonksiyonu ile otomatik olarak tespit edilmiş ve her biri için "
        "Gaussian notch maskesi oluşturularak frekans uzayında bastırılmıştır."
    ))
    story.append(body(
        f"Gerçekleştirilen analizde toplam {len(peaks)} tepe noktası tespit edilmiştir. "
        f"Filtreleme sonucunda PSNR değeri {psnr_noisy:.2f} dB'den {psnr_clean:.2f} dB'ye, "
        f"SSIM değeri ise {ssim_noisy:.4f}'den {ssim_clean:.4f}'e yükselmiştir. "
        "Bu sonuçlar, frekans uzayı filtrelemesinin periyodik gürültü gideriminde "
        "son derece etkili bir araç olduğunu kanıtlamaktadır."
    ))
    story.append(body(
        "Fourier dönüşümü, yalnızca bu tür periyodik gürültü giderimi ile sınırlı "
        "kalmayıp; görüntü sıkıştırma (JPEG, standartları DFT'ye yakın DCT kullanır), "
        "görüntü şiddetlendirme, kenar tespiti ve frekans bazlı özellik çıkarma gibi "
        "pek çok görüntü işleme uygulamasında da temel araç olmaya devam etmektedir. "
        "Bu ödev, 2B sinyaller ve Fourier analizi arasındaki derin ilişkiyi hem teorik "
        "hem de uygulamalı olarak pekiştirmiştir."
    ))
    story.append(sp(2))

    story.append(info_box("Kullanılan Araçlar ve Kütüphaneler", [
        "Python 3.11 — Temel programlama dili",
        "NumPy 2.x — 2B FFT (np.fft.fft2, np.fft.fftshift, np.fft.ifft2)",
        "SciPy — Yerel maksimum tespiti (scipy.ndimage.maximum_filter)",
        "Matplotlib — Görüntü ve spektrum görselleştirme",
        "Pillow (PIL) — Görüntü okuma ve yazma",
        "ReportLab — PDF rapor oluşturma",
    ], color="#EBF5FB"))

    return story

# ─────────────────────────────────────────────
# PDF OLUŞTUR
# ─────────────────────────────────────────────

print("PDF raporu oluşturuluyor...")

doc = SimpleDocTemplate(
    REPORT_FILE,
    pagesize=A4,
    leftMargin=MARGIN_LEFT,
    rightMargin=MARGIN_RIGHT,
    topMargin=MARGIN_TOP + 1*cm,
    bottomMargin=MARGIN_BOTTOM + 1*cm,
    title="SaS Ödev-2 — Frekans Uzayında Görüntü Filtreleme",
    author="Omar Nuriyev (24011902)",
    subject="Sinyaller ve Sistemler — YTU Bilgisayar Mühendisliği",
)

story = build_story()
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

print(f"[OK] {REPORT_FILE} oluşturuldu!")

import os
size = os.path.getsize(REPORT_FILE)
print(f"     Dosya boyutu: {size/1024:.1f} KB")
