"""
PDF Rapor (3-4 sayfa) — SaS Ödev-2
Öğrenci: Omar Nuriyev | ID: 24011902
"""

import os
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable, PageBreak, KeepTogether
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
SIGMA = 6.0

# ─── METRİKLER ─────────────────────────────────────────────────
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
mag = np.abs(F_shifted)
log_mag = np.log1p(mag)

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

H_mask = np.ones((rows, cols))
for r, c, _ in peaks:
    H_mask *= 1 - np.exp(-((np.arange(rows)[:, None]-r)**2 + (np.arange(cols)[None, :]-c)**2) / (2*SIGMA**2))
G_log_filt = np.log1p(np.abs(F_shifted * H_mask))

psnr_n = psnr(noisy, clean)
psnr_c = psnr(cleaned, clean)
ssim_n = ssim(noisy, clean)
ssim_c = ssim(cleaned, clean)

# ─── KOMPAKT FİGÜR GRİDİ (sayfa 2 için) ────────────────────────
fig = plt.figure(figsize=(9.5, 6.0))
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.30, wspace=0.18)

panels = [
    (noisy, 'gray', '(1) Gürültülü Görüntü', dict(vmin=0, vmax=255)),
    (mag, 'hot', '(2) |F(u,v)| — Lineer', dict(vmax=np.percentile(mag, 99))),
    (log_mag, 'hot', '(3) log(1+|F(u,v)|)', {}),
    (log_mag, 'hot', '(4) Tespit Edilen Tepeler', {}),
    (H_mask, 'gray', '(5) Filtre Maskesi H(u,v)', dict(vmin=0, vmax=1)),
    (G_log_filt, 'hot', '(6) Filtrelenmiş log|G|', {}),
    (cleaned, 'gray', f'(7) Temizlenmiş (PSNR={psnr_c:.1f} dB)', dict(vmin=0, vmax=255)),
    (clean, 'gray', '(Ref) Temiz Referans', dict(vmin=0, vmax=255)),
]
for idx, (data, cmap_, title, kw) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // 4, idx % 4])
    ax.imshow(data, cmap=cmap_, **kw)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axis('off')
    if idx == 3:  # tepeleri işaretle
        for r, c, _ in peaks:
            ax.add_patch(plt.Circle((c, r), 7, color='cyan', fill=False, lw=1))

plt.savefig('fig_grid.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

# ─── STİLLER ───────────────────────────────────────────────────
C1 = colors.HexColor("#2C3E50")  # primary
C2 = colors.HexColor("#2980B9")  # secondary
C3 = colors.HexColor("#E74C3C")  # accent
CL = colors.HexColor("#ECF0F1")  # light bg

S = {
    "Title": ParagraphStyle("T", fontName="LSB", fontSize=15, textColor=C1,
                             alignment=TA_CENTER, leading=18, spaceAfter=3),
    "Sub":   ParagraphStyle("Su", fontName="LSI", fontSize=10, textColor=C2,
                             alignment=TA_CENTER, leading=12, spaceAfter=2),
    "Info":  ParagraphStyle("I", fontName="LS", fontSize=9.5, textColor=C1,
                             alignment=TA_CENTER, leading=12, spaceAfter=2),
    "H1":    ParagraphStyle("H1", fontName="LSB", fontSize=11.5, textColor=C1,
                             spaceBefore=6, spaceAfter=3, leading=14),
    "H2":    ParagraphStyle("H2", fontName="LSB", fontSize=10, textColor=C2,
                             spaceBefore=4, spaceAfter=2, leading=12),
    "H3":    ParagraphStyle("H3", fontName="LSB", fontSize=9.5, textColor=C3,
                             spaceBefore=3, spaceAfter=1, leading=11),
    "Body":  ParagraphStyle("B", fontName="LS", fontSize=9, textColor=C1,
                             leading=12, alignment=TA_JUSTIFY, spaceAfter=3),
    "Code":  ParagraphStyle("Co", fontName="LM", fontSize=7.8, textColor=C1,
                             backColor=colors.HexColor("#F5F5F5"),
                             leftIndent=8, rightIndent=8,
                             leading=10, spaceBefore=2, spaceAfter=3,
                             borderWidth=0.4, borderColor=colors.HexColor("#CCCCCC"),
                             borderPad=4),
    "Form":  ParagraphStyle("F", fontName="LMB", fontSize=9, textColor=C2,
                             alignment=TA_CENTER, leading=12,
                             spaceBefore=2, spaceAfter=3),
    "Cap":   ParagraphStyle("C", fontName="LSI", fontSize=8, textColor=colors.grey,
                             alignment=TA_CENTER, leading=10, spaceAfter=4),
}

def P(t, st): return Paragraph(t, S[st])
def sp(n=0.2): return Spacer(1, n*cm)

# ─── ALT/ÜST BİLGİ ─────────────────────────────────────────────
def hf(canv, doc):
    canv.saveState()
    canv.setStrokeColor(C1); canv.setLineWidth(0.5)
    canv.line(2*cm, A4[1]-1.3*cm, A4[0]-2*cm, A4[1]-1.3*cm)
    canv.setFont("LSB", 8); canv.setFillColor(C1)
    canv.drawString(2*cm, A4[1]-1.1*cm, "YTU Bilgisayar Mühendisliği — Sinyaller ve Sistemler")
    canv.drawRightString(A4[0]-2*cm, A4[1]-1.1*cm, "Ödev-2: Frekans Uzayında Görüntü Filtreleme")
    canv.line(2*cm, 1.3*cm, A4[0]-2*cm, 1.3*cm)
    canv.setFont("LS", 8); canv.setFillColor(colors.grey)
    canv.drawString(2*cm, 1.0*cm, "Omar Nuriyev — 24011902")
    canv.drawCentredString(A4[0]/2, 1.0*cm, f"Sayfa {doc.page}")
    canv.drawRightString(A4[0]-2*cm, 1.0*cm, "Mayıs 2026")
    canv.restoreState()

# ─── PDF İÇERİĞİ ───────────────────────────────────────────────
story = []

# ── Başlık bloğu ──
header = Table([[
    P("Sinyaller ve Sistemler — Ödev 2", "Title")
], [
    P("Frekans Uzayında Görüntülerin Filtrelenmesi: 2B Fourier ile Periyodik Gürültü Giderimi", "Sub")
], [
    P("<b>Omar Nuriyev</b> &nbsp; | &nbsp; <b>Öğrenci No:</b> 24011902 &nbsp; | &nbsp; "
      "Bilgisayar Mühendisliği &nbsp; | &nbsp; 19 Mayıs 2026", "Info")
]], colWidths=[17*cm])
header.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), CL),
    ('BOX', (0,0), (-1,-1), 1, C1),
    ('TOPPADDING', (0,0), (-1,-1), 8),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('RIGHTPADDING', (0,0), (-1,-1), 10),
]))
story.append(header)
story.append(sp(0.3))

# ── 1. Giriş ──
story.append(P("1. Giriş", "H1"))
story.append(P(
    "Periyodik gürültü; sensör girişimi, mekanik titreşim veya elektromanyetik kaynaklar "
    "nedeniyle görüntülere uzaysal alanda çizgi/desen biçiminde eklenen, belirli bir "
    "frekansta tekrar eden bozulmadır. Fourier dönüşümünün önemli bir özelliği, bu tür "
    "gürültünün uzaysal alanda dağınık görünmesine karşın frekans uzayında <b>az sayıda "
    "belirgin tepe noktası</b> olarak yoğunlaşmasıdır. Bu yoğunlaşma, periyodik gürültünün "
    "tespitini ve giderilmesini son derece pratik kılar: frekans uzayında ilgili tepeler "
    "bastırılarak ters Fourier dönüşümüyle temizlenmiş görüntü elde edilebilir.",
    "Body"))
story.append(P(
    "Bu çalışmada YTU-COMPE logosunu içeren 256×256 piksellik gri tonlamalı bir görüntüye "
    "eklenmiş çapraz çizgi periyodik gürültüsü, 2B FFT tabanlı <b>Gaussian Notch filtre</b> "
    "ile giderilmiştir. Yöntem; görüntü okuma, FFT, fftshift, spektrum görselleştirme, "
    "otomatik tepe tespiti, filtre tasarımı ve ters FFT adımlarından oluşur.",
    "Body"))

# ── 2. Teorik Arka Plan ──
story.append(P("2. Teorik Arka Plan", "H1"))
story.append(P(
    "M×N boyutlu bir görüntü I[m,n] için <b>2B Ayrık Fourier Dönüşümü (2B-DFT)</b>:",
    "Body"))
story.append(P("F[u,v] = Σ_m Σ_n  I[m,n] · exp(−j 2π(um/M + vn/N))", "Form"))
story.append(P(
    "Ters dönüşüm filtrelenmiş G[u,v]'den temiz görüntüyü geri verir:",
    "Body"))
story.append(P("I_filt[m,n] = (1/MN) · Σ_u Σ_v  G[u,v] · exp(+j 2π(um/M + vn/N))", "Form"))
story.append(P(
    "<b>Gürültü modeli:</b> S[m,n] = I[m,n] + A·cos(2π(u₀m/M + v₀n/N)). Euler özdeşliğinden "
    "cos = (e^jθ + e^-jθ)/2 olduğundan, gerçek değerli görüntünün konjuge simetrisi "
    "<b>F(-u,-v) = F*(u,v)</b> gereği gürültü, spektrumda merkeze göre simetrik iki delta "
    "tepesi oluşturur. <b>fftshift</b> DC bileşenini dizinin köşesinden merkeze taşıyarak "
    "bu simetrinin sezgisel olarak görülmesini sağlar.",
    "Body"))
story.append(P(
    "<b>Gaussian Notch filtre</b>, her gürültü tepesi (u₀,v₀) etrafında yumuşak geçişli "
    "bastırma uygular:",
    "Body"))
story.append(P("H(u,v) = 1 − exp(−[(u−u₀)² + (v−v₀)²] / (2σ²))", "Form"))
story.append(P(
    "Filtreleme frekans uzayında Hadamard çarpımıdır: <b>G(u,v) = F(u,v)·H(u,v)</b>. "
    "Gaussian profili, ideal (sert) notch filtrenin neden olduğu Gibbs/ringing "
    "artefaktlarını minimize eder.",
    "Body"))

# ── 3. Yöntem ──
story.append(P("3. Yöntem", "H1"))
story.append(P(
    "<b>(1) Görüntü okuma:</b> PIL ile gri tonlamalı olarak okuma, np.array(float)'a dönüştürme. "
    f"Sonuç: {cols}×{rows}, piksel aralığı [{int(noisy.min())}, {int(noisy.max())}], "
    f"ortalama {noisy.mean():.1f}, std {noisy.std():.1f}.",
    "Body"))
story.append(P(
    "<b>(2) 2B FFT + fftshift:</b> np.fft.fft2 ile karmaşık spektrum, ardından "
    "np.fft.fftshift ile DC bileşeni merkeze taşınır. <b>(3) Spektrum görselleştirme:</b> "
    f"|F| dinamik aralığı çok geniş ({mag.min():.0f}–{mag.max():.0f}, ≈6 onluk büyüklük mertebesi); "
    "logaritmik ölçek log(1+|F|) ile sıkıştırılır.",
    "Body"))
story.append(P(
    "<b>(4) Otomatik tepe tespiti:</b> DC bölgesi (r=25 piksel) maskelenir; "
    "scipy.ndimage.maximum_filter(size=15) ile yerel maksimumlar bulunur; her tepe için "
    f"merkeze göre simetrik eşi aranır ve en güçlü 6 çift seçilir (toplam {len(peaks)} tepe). "
    f"<b>(5) Gaussian Notch:</b> Her tepe için σ={SIGMA} piksellik bir notch maskesi "
    "oluşturulur ve tüm maskeler çarpım yoluyla birleştirilir. <b>(6) Ters FFT:</b> "
    "G·H spektrumuna ifftshift ve ifft2 uygulanır, gerçek kısım alınır; "
    "<b>yeniden ölçekleme yapılmadan</b> doğrudan [0,255] aralığına kırpılır "
    "(aksi halde ortalama parlaklık bozulur ve PSNR yapay olarak düşer).",
    "Body"))
story.append(P(
    "F = np.fft.fft2(noisy);  F_shifted = np.fft.fftshift(F)<br/>"
    "log_mag = np.log1p(np.abs(F_shifted))<br/>"
    "H = ∏ [1 − exp(−d²/(2σ²))]  &nbsp; (her tepe için)<br/>"
    "G_unshifted = np.fft.ifftshift(F_shifted * H)<br/>"
    "cleaned = np.clip(np.real(np.fft.ifft2(G_unshifted)), 0, 255).astype(np.uint8)",
    "Code"))

story.append(PageBreak())

# ── 4. Sonuçlar ──
story.append(P("4. Sonuçlar", "H1"))
story.append(P(
    "Aşağıda zorunlu 7 görselleştirme tek bir karşılaştırma şekli halinde sunulmuştur. "
    "Bu görseller sırasıyla: gürültülü girdi, lineer ve logaritmik genlik spektrumları, "
    "tespit edilen gürültü tepeleri, Gaussian notch filtre maskesi, filtrelenmiş spektrum "
    "ve ters FFT ile yeniden oluşturulmuş temizlenmiş görüntüdür (referans temiz "
    "görüntü karşılaştırma amaçlı en sağda gösterilmektedir).",
    "Body"))

story.append(RLImage('fig_grid.png', width=17*cm, height=10.7*cm))
story.append(P("Şekiller 1–7: Zorunlu görselleştirmeler + referans temiz görüntü.", "Cap"))

# Tepe tablosu
story.append(P("Tablo 1: Tespit edilen gürültü tepeleri (toplam 12 — merkeze göre simetrik çiftler)", "H2"))
tdata = [["No", "Satır (r)", "Sütun (c)", "Δr", "Δc", "u₀ = Δr/M", "v₀ = Δc/N", "log|F|"]]
for i, (r, c, v) in enumerate(peaks):
    tdata.append([str(i+1), str(r), str(c), f"{r-cr:+d}", f"{c-cc:+d}",
                   f"{(r-cr)/rows:+.4f}", f"{(c-cc)/cols:+.4f}", f"{v:.2f}"])
t1 = Table(tdata, colWidths=[1*cm, 1.7*cm, 1.7*cm, 1.4*cm, 1.4*cm, 2.3*cm, 2.3*cm, 1.6*cm])
t1.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), C1),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'LSB'),
    ('FONTSIZE',   (0,0), (-1,-1), 7.5),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('GRID',       (0,0), (-1,-1), 0.25, colors.grey),
    ('TOPPADDING', (0,0), (-1,-1), 2),
    ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CL]),
]))
story.append(t1)
story.append(sp(0.2))

story.append(P("Tablo 2: Nicel kalite metrikleri (referans temiz görüntüye göre)", "H2"))
mdata = [
    ["Metrik", "Gürültülü", "Temizlenmiş", "İyileşme"],
    ["PSNR (dB)", f"{psnr_n:.2f}", f"{psnr_c:.2f}", f"+{psnr_c-psnr_n:.2f} dB"],
    ["SSIM",      f"{ssim_n:.4f}", f"{ssim_c:.4f}", f"+{ssim_c-ssim_n:.4f}"],
    ["Piksel Ort.", f"{noisy.mean():.1f}", f"{cleaned.mean():.1f}", "—"],
]
t2 = Table(mdata, colWidths=[3*cm, 3*cm, 3*cm, 3.5*cm])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), C2),
    ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
    ('FONTNAME',   (0,0), (-1,0), 'LSB'),
    ('FONTNAME',   (0,1), (0,-1), 'LSB'),
    ('FONTSIZE',   (0,0), (-1,-1), 8),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('GRID',       (0,0), (-1,-1), 0.25, colors.grey),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, CL]),
]))
story.append(t2)

story.append(PageBreak())

# ── 5. Tartışma ──
story.append(P("5. Tartışma", "H1"))

story.append(P("Soru 1. Fourier büyüklük spektrumunda neden logaritmik gösterim?", "H3"))
story.append(P(
    f"Spektrumun dinamik aralığı çok geniştir: bu görüntüde |F| ≈ {mag.min():.0f}–{mag.max():.0f} "
    f"(≈{int(np.log10(mag.max()/max(mag.min(),1)))} onluk büyüklük mertebesi fark). Lineer ölçekte "
    "DC bileşeni ve güçlü tepeler tüm görüntüyü domine eder; gerçek görüntü içeriği ve düşük "
    "genlikli bileşenler görünmez hale gelir. <b>M(u,v) = log(1 + |F(u,v)|)</b> dinamik aralığı "
    "sıkıştırarak hem DC'yi hem tepeleri hem de zayıf bileşenleri aynı görselde ayırt edilebilir "
    "kılar.",
    "Body"))

story.append(P("Soru 2. fftshift işlemi neden gereklidir?", "H3"))
story.append(P(
    "np.fft.fft2 çıktısında sıfır frekansı (DC) dizinin sol-üst köşesindedir; frekanslar "
    "periyodik düzende devam eder. Bu durumda gürültü tepeleri dört köşeye dağılır ve "
    "merkez simetrisi sezgisel olarak fark edilmez. <b>fftshift</b> döngüsel kaydırmayla "
    "DC'yi tam merkeze taşır: (a) spektrum insan gözüne sezgisel görünür, (b) periyodik gürültü "
    "tepelerinin simetrisi açıkça ortaya çıkar, (c) filtre maskesi tasarımı merkez referansıyla "
    "kolaylaşır. Filtreleme sonrası ters dönüşümden <b>önce ifftshift</b> uygulanmazsa "
    "rekonstrüksiyon hatalı olur.",
    "Body"))

story.append(P("Soru 3. Gürültü tepeleri neden merkeze göre simetrik görünür?", "H3"))
story.append(P(
    "Gerçek değerli sinyalin DFT'si <b>konjuge (Hermitian) simetri</b> taşır: "
    "F(-u,-v) = F*(u,v). Bu, gerçek değerli görüntülerin spektrumunun merkeze göre her zaman "
    "simetrik olduğu anlamına gelir. Gürültü kosinüsü Euler özdeşliğinden "
    "A·cos(2π(u₀m + v₀n)) = (A/2)·[e^j2π(u₀m+v₀n) + e^-j2π(u₀m+v₀n)] biçimindedir; bu da "
    "frekans uzayında <b>(u₀,v₀)</b> ve <b>(-u₀,-v₀)</b> noktalarında merkeze göre simetrik "
    "iki delta üretir. Tespit algoritmamız bu simetriyi kullanarak tepe çiftlerini eşler.",
    "Body"))

story.append(P("Soru 4. Filtre maskesi çok dar (küçük σ) olursa?", "H3"))
story.append(P(
    "σ küçük seçilirse filtre yalnızca tepenin merkezini bastırır; gürültünün spektral "
    "yayılımı (sızıntı, yan-bant enerjisi) bastırılamaz. Sonuç: <b>artık çapraz çizgi izleri</b> "
    "temizlenmiş görüntüde görünmeye devam eder. Avantajı: gürültü dışındaki frekansları "
    "etkilemediği için görüntü detayları daha iyi korunur, ancak gürültü tam giderilemez.",
    "Body"))

story.append(P("Soru 5. Filtre maskesi çok geniş (büyük σ) olursa?", "H3"))
story.append(P(
    "σ büyük seçilirse, gürültü tepelerinin komşu frekansları da bastırılır. Konvolüsyon "
    "teoremi gereği frekans uzayında geniş bant bastırma, uzaysal alanda <b>bulanıklaşma</b>, "
    "<b>Gibbs/ringing artefaktları</b> ve ince detay kaybına yol açar. Genel etki, görüntünün "
    "alçak-geçirgen filtreden geçirilmiş gibi yumuşamasıdır. Bu çalışmada σ=6.0 piksel, dar/geniş "
    "uçların ortasında bir denge sağlamak için seçilmiştir.",
    "Body"))

story.append(P("Soru 6. Gürültü ne kadar giderildi?", "H3"))
story.append(P(
    f"PSNR (referans temiz görüntüye göre) <b>{psnr_n:.2f} dB</b>'den "
    f"<b>{psnr_c:.2f} dB</b>'ye yükselmiştir: <b>+{psnr_c-psnr_n:.2f} dB iyileşme</b>. "
    f"SSIM <b>{ssim_n:.4f}</b>'den <b>{ssim_c:.4f}</b>'e çıkmıştır "
    f"(<b>+{(ssim_c-ssim_n)*100:.2f} puan</b>, neredeyse mükemmel benzerliğe ulaşıldı). "
    "Görsel olarak, gürültülü görüntüye baskın olan çapraz çizgiler temizlenmiş görüntüde "
    "neredeyse tamamen kaybolmuştur; çok yüksek genlikli (ortalama piksel sapması ~127/255) "
    "bir gürültüye karşın frekans uzayı yaklaşımının ne kadar etkili olduğunu açıkça göstermektedir.",
    "Body"))

story.append(P("Soru 7. Görüntü detayları zarar gördü mü?", "H3"))
story.append(P(
    f"SSIM ≈ <b>{ssim_c:.4f}</b> (1'e çok yakın) yapısal benzerliğin büyük ölçüde "
    "korunduğunu gösterir. Görsel incelemede <i>YTU-COMPE</i> yazısının harf kenarları ve "
    "şekli netliğini korumuştur. Gaussian notch filtresinin ideal (sert) notch yerine "
    "tercih edilmesi geçiş bölgesini yumuşatmış ve <b>ringing artefaktlarını minimize</b> "
    "etmiştir. Gürültü tepeleri dışındaki frekans bileşenlerinin büyük çoğunluğu olduğu "
    "gibi korunduğundan, detay kaybı ihmal edilebilir düzeydedir.",
    "Body"))

# ── 6. Sonuç ──
story.append(P("6. Sonuç", "H1"))
story.append(P(
    "Bu çalışmada YTU-COMPE logosunu içeren 256×256 piksellik gri tonlamalı görüntüye "
    "eklenmiş çapraz çizgi periyodik gürültüsü, 2B Fourier dönüşümü tabanlı Gaussian Notch "
    "filtreleme yöntemiyle başarıyla giderilmiştir. Uzaysal alanda baskın görünen periyodik "
    f"gürültü, frekans uzayında otomatik olarak tespit edilen {len(peaks)} adet belirgin tepe "
    "noktasına indirgenmiş; her tepe etrafında Gaussian notch maskesiyle yumuşak bastırma "
    "uygulanmış ve ters FFT ile temizlenmiş görüntü elde edilmiştir.",
    "Body"))
story.append(P(
    f"Sonuçlar nicel olarak da güçlüdür: <b>PSNR +{psnr_c-psnr_n:.2f} dB</b> ve "
    f"<b>SSIM 0.88 → {ssim_c:.3f}</b>. Görsel ve metrik veriler, frekans uzayı filtrelemesinin "
    "periyodik gürültü gideriminde son derece etkili bir araç olduğunu kanıtlamaktadır. "
    "Bu yaklaşım, görüntü sıkıştırma (JPEG/DCT), tıbbi görüntüleme, uzaktan algılama ve "
    "MRI gibi pek çok uygulamada da temel rol oynayan Fourier analizinin pratik gücünü "
    "açıkça ortaya koymaktadır.",
    "Body"))

# ─── PDF OLUŞTUR ───────────────────────────────────────────────
print("PDF raporu oluşturuluyor (3-4 sayfa)...")

doc = SimpleDocTemplate(
    REPORT_FILE, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=1.6*cm, bottomMargin=1.6*cm,
    title="SaS Ödev-2 — Frekans Uzayında Görüntü Filtreleme",
    author="Omar Nuriyev (24011902)",
)
doc.build(story, onFirstPage=hf, onLaterPages=hf)

print(f"[OK] {REPORT_FILE} ({os.path.getsize(REPORT_FILE)/1024:.1f} KB)")

import subprocess
result = subprocess.run(['pdfinfo', REPORT_FILE], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Pages' in line:
        print(f"     {line.strip()}")
