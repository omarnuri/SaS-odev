"""
TEK SAYFA PDF Rapor — SaS Ödev-2
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
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, KeepInFrame
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# Liberation Sans (Türkçe karakter desteği)
_F = "/usr/share/fonts/truetype/liberation"
pdfmetrics.registerFont(TTFont("LS",   f"{_F}/LiberationSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LSB",  f"{_F}/LiberationSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("LSI",  f"{_F}/LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("LSBI", f"{_F}/LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("LM",   f"{_F}/LiberationMono-Regular.ttf"))
pdfmetrics.registerFont(TTFont("LMB",  f"{_F}/LiberationMono-Bold.ttf"))
registerFontFamily("LS", normal="LS", bold="LSB", italic="LSI", boldItalic="LSBI")

REPORT_FILE = "Sas_24011902.pdf"

# ─── METRİKLERİ HESAPLA ────────────────────────────────────────
def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 20*np.log10(255/np.sqrt(mse)) if mse > 0 else float('inf')

def ssim(a, b):
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    a, b = a.astype(float), b.astype(float)
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.var(), b.var()
    s12 = np.mean((a - mu1) * (b - mu2))
    return ((2*mu1*mu2+c1)*(2*s12+c2)) / ((mu1**2+mu2**2+c1)*(s1+s2+c2))

noisy = np.array(Image.open("assignment/noisy_image.png").convert("L"), dtype=float)
clean = np.array(Image.open("assignment/image.png").convert("L"), dtype=float)
cleaned = np.array(Image.open("cleaned_image.png").convert("L"), dtype=float)
rows, cols = noisy.shape
cr, cc = rows//2, cols//2

F = np.fft.fft2(noisy)
F_shifted = np.fft.fftshift(F)
log_mag = np.log1p(np.abs(F_shifted))

work = log_mag.copy()
Y, X = np.ogrid[:rows, :cols]
work[(Y-cr)**2 + (X-cc)**2 <= 25**2] = 0
loc_max = maximum_filter(work, size=15)
pr_, pc_ = np.where((work == loc_max) & (work > 0))
pv_ = log_mag[pr_, pc_]
order = np.argsort(pv_)[::-1]
peaks = []
used = set()
for i in order:
    if i in used: continue
    r, c = pr_[i], pc_[i]
    sr, sc = 2*cr-r, 2*cc-c
    for j in order:
        if j == i or j in used: continue
        if abs(pr_[j]-sr) <= 3 and abs(pc_[j]-sc) <= 3:
            peaks.append((r, c, pv_[i]))
            peaks.append((pr_[j], pc_[j], pv_[j]))
            used.add(i); used.add(j); break
    if len(peaks) >= 12: break

psnr_n = psnr(noisy, clean)
psnr_c = psnr(cleaned, clean)
ssim_n = ssim(noisy, clean)
ssim_c = ssim(cleaned, clean)

# ─── STİLLER (kompakt) ─────────────────────────────────────────
C_PRIMARY = colors.HexColor("#2C3E50")
C_SEC     = colors.HexColor("#2980B9")
C_ACC     = colors.HexColor("#E74C3C")
C_LIGHT   = colors.HexColor("#ECF0F1")

S_TITLE = ParagraphStyle("T", fontName="LSB", fontSize=11, textColor=C_PRIMARY,
                          alignment=TA_CENTER, leading=13, spaceAfter=1)
S_SUB   = ParagraphStyle("Su", fontName="LSI", fontSize=7.5, textColor=C_SEC,
                          alignment=TA_CENTER, leading=9, spaceAfter=2)
S_H     = ParagraphStyle("H", fontName="LSB", fontSize=7.5, textColor=C_PRIMARY,
                          spaceBefore=1, spaceAfter=1, leading=9)
S_BODY  = ParagraphStyle("B", fontName="LS", fontSize=6.5, textColor=C_PRIMARY,
                          leading=8, alignment=TA_JUSTIFY, spaceAfter=1)
S_CAP   = ParagraphStyle("C", fontName="LSI", fontSize=5.5, textColor=colors.grey,
                          alignment=TA_CENTER, leading=7, spaceAfter=0)
S_QA    = ParagraphStyle("Q", fontName="LS", fontSize=6.2, textColor=C_PRIMARY,
                          leading=7.5, alignment=TA_JUSTIFY, spaceAfter=0)
S_FORM  = ParagraphStyle("F", fontName="LMB", fontSize=6.5, textColor=C_SEC,
                          alignment=TA_CENTER, leading=8, spaceAfter=1, spaceBefore=1)

def P(text, style): return Paragraph(text, style)

# ─── FIGÜRLERI TEK GRID OLARAK ÜRET ────────────────────────────
import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(9.5, 4.4))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.15)

ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(noisy, cmap='gray', vmin=0, vmax=255)
ax1.set_title('(1) Gürültülü', fontsize=8, fontweight='bold'); ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(np.abs(F_shifted), cmap='hot', vmax=np.percentile(np.abs(F_shifted), 99))
ax2.set_title('(2) |F(u,v)| lineer', fontsize=8, fontweight='bold'); ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(log_mag, cmap='hot')
ax3.set_title('(3) log(1+|F|)', fontsize=8, fontweight='bold'); ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
ax4.imshow(log_mag, cmap='hot')
for r, c, _ in peaks:
    ax4.add_patch(plt.Circle((c, r), 6, color='cyan', fill=False, lw=0.8))
ax4.set_title('(4) Tespit edilen tepeler', fontsize=8, fontweight='bold'); ax4.axis('off')

# Maskeyi yeniden hesapla
H_mask = np.ones((rows, cols))
for r, c, _ in peaks:
    H_mask *= 1 - np.exp(-((np.arange(rows)[:, None]-r)**2 + (np.arange(cols)[None, :]-c)**2) / (2*6.0**2))
G_log_filt = np.log1p(np.abs(F_shifted * H_mask))

ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(H_mask, cmap='gray', vmin=0, vmax=1)
ax5.set_title('(5) Filtre maskesi H(u,v)', fontsize=8, fontweight='bold'); ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(G_log_filt, cmap='hot')
ax6.set_title('(6) Filtrelenmiş log(|G|)', fontsize=8, fontweight='bold'); ax6.axis('off')

ax7 = fig.add_subplot(gs[1, 2])
ax7.imshow(cleaned, cmap='gray', vmin=0, vmax=255)
ax7.set_title(f'(7) Temizlenmiş (PSNR={psnr_c:.1f} dB)', fontsize=8, fontweight='bold'); ax7.axis('off')

ax8 = fig.add_subplot(gs[1, 3])
ax8.imshow(clean, cmap='gray', vmin=0, vmax=255)
ax8.set_title('Referans temiz', fontsize=8, fontweight='bold'); ax8.axis('off')

plt.savefig('fig_onepage_grid.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ─── PDF (TEK SAYFA) ───────────────────────────────────────────
print("Tek sayfa PDF oluşturuluyor...")

doc = SimpleDocTemplate(
    REPORT_FILE, pagesize=A4,
    leftMargin=1.0*cm, rightMargin=1.0*cm,
    topMargin=0.8*cm, bottomMargin=0.7*cm,
    title="SaS Ödev-2 — Frekans Uzayında Görüntü Filtreleme",
    author="Omar Nuriyev (24011902)",
)

story = []

# Başlık bloğu
header = Table([[
    P("<b>Sinyaller ve Sistemler — Ödev 2: Frekans Uzayında Görüntü Filtreleme</b>", S_TITLE),
], [
    P("<b>Omar Nuriyev — 24011902</b> | YTU Bilgisayar Mühendisliği | 19 Mayıs 2026", S_SUB)
]], colWidths=[19*cm])
header.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), C_LIGHT),
    ('BOX', (0,0), (-1,-1), 0.5, C_PRIMARY),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(header)
story.append(Spacer(1, 0.15*cm))

# Giriş / Yöntem — kısa
intro_text = (
    "<b>Amaç:</b> 256×256 piksellik gri tonlamalı YTU-COMPE görüntüsüne eklenmiş çapraz çizgi "
    "periyodik gürültüyü 2B FFT tabanlı Gaussian notch filtre ile gidermek. <b>Gürültü modeli:</b> "
    "S(x,y) = I(x,y) + A·cos(2π(u₀x+v₀y)). Bu gürültü, gerçek-değerli sinyalin konjuge simetrisi "
    "gereği frekans uzayında merkeze göre simetrik iki delta tepesi oluşturur."
)
story.append(P(intro_text, S_BODY))

# Yöntem adımları
method_text = (
    "<b>Yöntem (6 adım):</b> "
    "<b>(1)</b> PIL ile gri tonlamalı okuma → np.array(float). "
    "<b>(2)</b> F = np.fft.fft2(noisy); F_shifted = np.fft.fftshift(F) — DC merkeze. "
    "<b>(3)</b> log(1+|F|) ile dinamik aralık sıkıştırılır (≈6 onluk büyüklük mertebesi). "
    "<b>(4)</b> DC bölgesini maskele (r=25), scipy.ndimage.maximum_filter ile yerel maksimumlar, simetrik çiftler eşle (12 tepe). "
    "<b>(5)</b> Her tepe için Gaussian notch maskesi: H = ∏ [1 − exp(−d²/(2σ²))], σ=6.0 piksel. "
    "<b>(6)</b> G = F_shifted·H → ifftshift → np.real(ifft2(·)) → np.clip([0,255]) — <i>yeniden ölçekleme yapılmaz</i>, "
    "aksi halde ortalama parlaklık bozulur."
)
story.append(P(method_text, S_BODY))

# 8'li figür grid
story.append(Spacer(1, 0.1*cm))
fig_img = RLImage('fig_onepage_grid.png', width=19*cm, height=8.3*cm)
story.append(fig_img)
story.append(P("Şekiller 1–7: zorunlu görselleştirmeler + referans temiz görüntü karşılaştırması.", S_CAP))

# Sonuçlar tablosu (yatay, kompakt)
results = [
    ["", "Gürültülü", "Temizlenmiş", "İyileşme"],
    ["PSNR (dB)", f"{psnr_n:.2f}", f"{psnr_c:.2f}", f"+{psnr_c-psnr_n:.2f} dB"],
    ["SSIM",      f"{ssim_n:.4f}", f"{ssim_c:.4f}", f"+{ssim_c-ssim_n:.4f}"],
]
# İlk 6 tepe (yerden tasarruf için)
peak_rows = [["#", "Satır", "Sütun", "Δr", "Δc", "u₀=Δr/M", "v₀=Δc/N", "log|F|"]]
for i, (r, c, v) in enumerate(peaks[:6]):
    peak_rows.append([str(i+1), str(r), str(c), f"{r-cr:+d}", f"{c-cc:+d}",
                       f"{(r-cr)/rows:+.4f}", f"{(c-cc)/cols:+.4f}", f"{v:.2f}"])

t_metrics = Table(results, colWidths=[2.4*cm, 1.8*cm, 1.9*cm, 2.3*cm])
t_metrics.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), C_SEC),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'LSB'),
    ('FONTNAME',   (0,1), (0,-1), 'LSB'),
    ('FONTSIZE',   (0,0), (-1,-1), 6.5),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('GRID',       (0,0), (-1,-1), 0.25, colors.grey),
    ('TOPPADDING', (0,0), (-1,-1), 1.5),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_LIGHT]),
]))

t_peaks = Table(peak_rows, colWidths=[0.6*cm, 1.0*cm, 1.0*cm, 0.8*cm, 0.8*cm, 1.6*cm, 1.6*cm, 1.0*cm])
t_peaks.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'LSB'),
    ('FONTSIZE',   (0,0), (-1,-1), 5.8),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('GRID',       (0,0), (-1,-1), 0.25, colors.grey),
    ('TOPPADDING', (0,0), (-1,-1), 1),
    ('BOTTOMPADDING', (0,0), (-1,-1), 1),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_LIGHT]),
]))

# Yan yana tablolar
tables_row = Table([[
    [P("<b>Tablo 1: Kalite metrikleri</b>", S_H), t_metrics],
    [P("<b>Tablo 2: Tespit edilen tepe noktaları (ilk 6, toplam 12 — kalanlar simetrik)</b>", S_H), t_peaks]
]], colWidths=[8.4*cm, 10.5*cm])
tables_row.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 0),
    ('RIGHTPADDING', (0,0), (-1,-1), 0),
]))
story.append(Spacer(1, 0.1*cm))
story.append(tables_row)

# Tartışma — kısa Q&A
story.append(Spacer(1, 0.1*cm))
story.append(P("<b>Tartışma (7 soru):</b>", S_H))

qa = [
    ("<b>Q1. Neden logaritmik spektrum?</b>",
     f"Dinamik aralık çok geniş (|F|: {np.abs(F_shifted).min():.0f}–{np.abs(F_shifted).max():.0f}, ~6 mertebe). log(1+|F|) sıkıştırması olmadan DC dışındaki bileşenler görünmez."),
    ("<b>Q2. Neden fftshift gerekli?</b>",
     "fft2 çıktısında DC köşelerdedir; periyodik gürültü tepeleri 4 köşeye dağılır. fftshift DC'yi merkeze taşır, simetri görünür hale gelir, filtre tasarımı kolaylaşır. Ters dönüşümden önce ifftshift mutlaka uygulanmalı."),
    ("<b>Q3. Neden simetrik tepeler?</b>",
     "Gerçek değerli sinyal için F(-u,-v) = F*(u,v) (Hermitian simetri). cos(θ) = (e^jθ+e^-jθ)/2 olduğundan tek bir kosinüs gürültü merkeze göre simetrik iki delta tepesi üretir."),
    ("<b>Q4. Çok dar filtre (küçük σ)?</b>",
     "Tepenin yalnızca merkezi bastırılır; gürültünün yan-bant enerjisi kalır → çapraz çizgiler kısmen sürer. Detay korunur ama gürültü tam giderilmez."),
    ("<b>Q5. Çok geniş filtre (büyük σ)?</b>",
     "Tepenin komşu frekansları da bastırılır → uzaysal alanda bulanıklaşma ve Gibbs/ringing artefaktları, ince detay kaybı (alçak geçirgen filtreye benzer etki)."),
    ("<b>Q6. Gürültü ne kadar giderildi?</b>",
     f"PSNR {psnr_n:.2f} dB → {psnr_c:.2f} dB (<b>+{psnr_c-psnr_n:.2f} dB</b>). SSIM {ssim_n:.4f} → {ssim_c:.4f} (<b>+{(ssim_c-ssim_n)*100:.1f} puan</b>). Çapraz çizgiler görsel olarak ortadan kalktı."),
    ("<b>Q7. Detay bozuldu mu?</b>",
     f"SSIM ≈ {ssim_c:.3f} (1'e çok yakın) → yapısal benzerlik korundu. YTU-COMPE yazısı net. Gaussian notch yumuşak geçiş sağladığı için ringing minimum; gürültü tepeleri dışındaki bilgi büyük ölçüde korundu."),
]

# 2 sütun layout için Q&A'yi tabloya koy
left_qa, right_qa = [], []
for i, (q, a) in enumerate(qa):
    cell = [P(q, S_QA), P(a, S_QA), Spacer(1, 0.05*cm)]
    (left_qa if i < 4 else right_qa).append(cell)

qa_table = Table([[left_qa, right_qa]], colWidths=[9.4*cm, 9.4*cm])
qa_table.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 2),
    ('RIGHTPADDING', (0,0), (-1,-1), 2),
]))
story.append(qa_table)

# Sonuç
story.append(Spacer(1, 0.05*cm))
story.append(P(
    "<b>Sonuç:</b> 2B FFT + 12 noktada Gaussian notch (σ=6.0) ile çapraz çizgi periyodik gürültü "
    f"başarıyla giderildi: PSNR +{psnr_c-psnr_n:.2f} dB, SSIM +{(ssim_c-ssim_n)*100:.1f} puan. "
    "Frekans uzayı yaklaşımı, uzaysal alanda baskın görünen gürültüyü yalnızca birkaç frekans bileşenine "
    "indirgediği için periyodik gürültü gideriminde son derece etkili bir araçtır.",
    S_BODY))

# KeepInFrame ile tek sayfaya zorla
final = KeepInFrame(19*cm, 28.5*cm, story, mode='shrink')
doc.build([final])

import os
print(f"[OK] {REPORT_FILE} ({os.path.getsize(REPORT_FILE)/1024:.1f} KB)")

# Sayfa sayısını doğrula
import subprocess
result = subprocess.run(['pdfinfo', REPORT_FILE], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Pages' in line:
        print(f"     {line.strip()}")
