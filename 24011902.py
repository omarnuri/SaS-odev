"""
SaS Ödev-2: Frekans Uzayında Periyodik Gürültü Giderimi
Öğrenci: Omar Nuriyev | ID: 24011902
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
import os

# ─────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def psnr(img1, img2):
    """Peak Signal-to-Noise Ratio (dB)"""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim(img1, img2):
    """Structural Similarity Index (simplified)"""
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
    """Dairesel maske oluştur: merkez içinde True"""
    rows, cols = shape
    cy, cx = center
    Y, X = np.ogrid[:rows, :cols]
    return (Y - cy) ** 2 + (X - cx) ** 2 <= radius ** 2

def gaussian_notch(shape, center, sigma):
    """Gaussian notch: merkezdeki frekansları bastır (H=0 merkez, H=1 uzak)"""
    rows, cols = shape
    cy, cx = center
    Y, X = np.ogrid[:rows, :cols]
    dist2 = (Y - cy) ** 2 + (X - cx) ** 2
    return 1.0 - np.exp(-dist2 / (2 * sigma ** 2))

def find_noise_peaks(log_mag, dc_radius=25, local_size=15, n_pairs=4):
    """
    Log-magnitude spektrumunda gürültü tepelerini bul.
    DC bileşenini dışla, yerel maksimumları bul, en güçlü simetrik çiftleri döndür.
    """
    rows, cols = log_mag.shape
    crow, ccol = rows // 2, cols // 2

    # DC bölgesini maskele
    work = log_mag.copy()
    dc_mask = circular_mask(log_mag.shape, (crow, ccol), dc_radius)
    work[dc_mask] = 0

    # Yerel maksimumları bul
    local_max = maximum_filter(work, size=local_size)
    is_peak = (work == local_max) & (work > 0)

    # Tepe koordinatları ve değerleri
    peak_rows, peak_cols = np.where(is_peak)
    peak_vals = log_mag[peak_rows, peak_cols]

    # Değere göre sırala
    order = np.argsort(peak_vals)[::-1]
    peak_rows = peak_rows[order]
    peak_cols = peak_cols[order]
    peak_vals = peak_vals[order]

    # Simetrik çiftleri seç: (r,c) ve (2*crow-r, 2*ccol-c) birlikte var mı?
    selected = []
    used = set()
    for i in range(len(peak_rows)):
        if i in used:
            continue
        r, c = peak_rows[i], peak_cols[i]
        sym_r, sym_c = 2 * crow - r, 2 * ccol - c
        # Simetrik çifti ara
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

    # Eğer simetrik çift bulunamadıysa sadece en güçlü tepeleri al
    if len(selected) == 0:
        for i in range(min(n_pairs * 2, len(peak_rows))):
            selected.append((peak_rows[i], peak_cols[i], peak_vals[i]))

    return selected


# ─────────────────────────────────────────────
# ADIM 1: GÖRÜNTÜ OKUMA VE GÖRÜNTÜLEME
# ─────────────────────────────────────────────
print("=" * 60)
print("SaS Ödev-2: Frekans Uzayında Görüntü Filtreleme")
print("Öğrenci: Omar Nuriyev | ID: 24011902")
print("=" * 60)

img_path = "assignment/noisy_image.png"
clean_path = "assignment/image.png"

noisy_pil = Image.open(img_path)
if noisy_pil.mode != 'L':
    noisy_pil = noisy_pil.convert('L')
noisy = np.array(noisy_pil, dtype=float)

clean_pil = Image.open(clean_path)
if clean_pil.mode != 'L':
    clean_pil = clean_pil.convert('L')
clean_ref = np.array(clean_pil, dtype=float)

rows, cols = noisy.shape
print(f"\nGörüntü boyutu       : {cols}x{rows} piksel (Genişlik x Yükseklik)")
print(f"Renk modu            : Gri tonlamalı (8-bit)")
print(f"Piksel değer aralığı : [{noisy.min():.0f}, {noisy.max():.0f}]")
print(f"Ortalama piksel      : {noisy.mean():.2f}")
print(f"Standart sapma       : {noisy.std():.2f}")

# Şekil 1: Gürültülü görüntü
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
im1 = ax1.imshow(noisy, cmap='gray', vmin=0, vmax=255)
ax1.set_title('Şekil 1: Gürültülü Görüntü\n(Çapraz Çizgi Periyodik Gürültü)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sütun (Piksel)')
ax1.set_ylabel('Satır (Piksel)')
plt.colorbar(im1, ax=ax1, label='Piksel Değeri')
plt.tight_layout()
plt.savefig('fig1_noisy.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] fig1_noisy.png kaydedildi.")


# ─────────────────────────────────────────────
# ADIM 2: 2B FOURIER DÖNÜŞÜMü
# ─────────────────────────────────────────────
print("\n--- ADIM 2: 2B FFT Hesaplama ---")

F = np.fft.fft2(noisy)
F_shifted = np.fft.fftshift(F)

print(f"FFT matris boyutu    : {F.shape}")
print(f"DC bileşen (merkez)  : |F(0,0)| = {np.abs(F_shifted[rows//2, cols//2]):.2f}")
print(f"Maksimum genlik      : {np.abs(F_shifted).max():.2f}")


# ─────────────────────────────────────────────
# ADIM 3: GENLIK SPEKTRUMU GÖRSELLEŞTIRME
# ─────────────────────────────────────────────
print("\n--- ADIM 3: Genlik Spektrumu ---")

magnitude = np.abs(F_shifted)
log_magnitude = np.log1p(magnitude)

print(f"Lineer genlik aralığı: [{magnitude.min():.2f}, {magnitude.max():.2f}]")
print(f"Log genlik aralığı   : [{log_magnitude.min():.4f}, {log_magnitude.max():.4f}]")

# Şekil 2: Lineer genlik spektrumu
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
im2 = ax2.imshow(magnitude, cmap='hot',
                 vmax=np.percentile(magnitude, 99))
ax2.set_title('Şekil 2: 2B FFT Lineer Genlik Spektrumu\n|F(u,v)|', fontsize=12, fontweight='bold')
ax2.set_xlabel('Frekans u')
ax2.set_ylabel('Frekans v')
ax2.axhline(rows//2, color='cyan', linewidth=0.5, alpha=0.5)
ax2.axvline(cols//2, color='cyan', linewidth=0.5, alpha=0.5)
plt.colorbar(im2, ax=ax2, label='Genlik')
plt.tight_layout()
plt.savefig('fig2_fft_linear.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig2_fft_linear.png kaydedildi.")

# Şekil 3: Logaritmik genlik spektrumu
fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
im3 = ax3.imshow(log_magnitude, cmap='hot')
ax3.set_title('Şekil 3: 2B FFT Logaritmik Genlik Spektrumu\nM(u,v) = log(1 + |F(u,v)|)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Frekans u')
ax3.set_ylabel('Frekans v')
ax3.axhline(rows//2, color='cyan', linewidth=0.5, alpha=0.5)
ax3.axvline(cols//2, color='cyan', linewidth=0.5, alpha=0.5)
plt.colorbar(im3, ax=ax3, label='log(1 + |F|)')
plt.tight_layout()
plt.savefig('fig3_fft_log.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig3_fft_log.png kaydedildi.")


# ─────────────────────────────────────────────
# ADIM 4: GÜRÜLTÜ FREKANSLARının TESPİTİ
# ─────────────────────────────────────────────
print("\n--- ADIM 4: Gürültü Frekanslarının Tespiti ---")

crow, ccol = rows // 2, cols // 2
peaks = find_noise_peaks(log_magnitude, dc_radius=25, local_size=15, n_pairs=6)

print(f"\nToplam {len(peaks)} tepe noktası tespit edildi:")
print(f"{'No':>3} | {'Satır':>6} | {'Sütun':>6} | {'Δsatır':>7} | {'Δsütun':>7} | {'u₀':>8} | {'v₀':>8} | {'Log-Genlik':>10}")
print("-" * 75)
for i, (r, c, val) in enumerate(peaks):
    dr = r - crow
    dc_off = c - ccol
    u0 = dr / rows
    v0 = dc_off / cols
    print(f"{i+1:>3} | {r:>6} | {c:>6} | {dr:>+7} | {dc_off:>+7} | {u0:>+8.4f} | {v0:>+8.4f} | {val:>10.3f}")

# Şekil 4: Tepe noktaları işaretli log-spektrum
fig4, ax4 = plt.subplots(1, 1, figsize=(7, 7))
im4 = ax4.imshow(log_magnitude, cmap='hot')
ax4.set_title('Şekil 4: Tespit Edilen Gürültü Frekans Tepeleri\n(Kırmızı: Tespit edilen tepeler)', fontsize=11, fontweight='bold')
ax4.set_xlabel('Frekans u')
ax4.set_ylabel('Frekans v')
ax4.axhline(crow, color='cyan', linewidth=0.5, alpha=0.4)
ax4.axvline(ccol, color='cyan', linewidth=0.5, alpha=0.4)
for i, (r, c, val) in enumerate(peaks):
    circ = plt.Circle((c, r), 8, color='red', fill=False, linewidth=1.5)
    ax4.add_patch(circ)
    ax4.annotate(f'P{i+1}', xy=(c, r), xytext=(c+10, r-10),
                 color='yellow', fontsize=7,
                 arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8))
plt.colorbar(im4, ax=ax4, label='log(1 + |F|)')
plt.tight_layout()
plt.savefig('fig4_peaks.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[OK] fig4_peaks.png kaydedildi.")


# ─────────────────────────────────────────────
# ADIM 5: FİLTRE MASKESI TASARIMI
# ─────────────────────────────────────────────
print("\n--- ADIM 5: Notch Filtre Tasarımı ---")

FILTER_SIGMA = 6.0  # Gaussian notch yarıçapı (piksel)

H = np.ones((rows, cols), dtype=float)
for r, c, _ in peaks:
    H *= gaussian_notch((rows, cols), (r, c), FILTER_SIGMA)

print(f"Filtre tipi          : Gaussian Notch")
print(f"Sigma (σ)            : {FILTER_SIGMA} piksel")
print(f"Bastırılan tepe sayısı: {len(peaks)}")
print(f"Filtre min değeri    : {H.min():.4f}")
print(f"Filtre max değeri    : {H.max():.4f}")

# Şekil 5: Filtre maskesi
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 5))
im5a = axes5[0].imshow(H, cmap='gray', vmin=0, vmax=1)
axes5[0].set_title('Şekil 5a: H(u,v) Filtre Maskesi\n(Siyah = bastırılan frekanslar)', fontsize=11, fontweight='bold')
axes5[0].set_xlabel('Frekans u')
axes5[0].set_ylabel('Frekans v')
plt.colorbar(im5a, ax=axes5[0], label='H(u,v)')

# Merkezin yakınını göster
zoom = 60
im5b = axes5[1].imshow(H[crow-zoom:crow+zoom, ccol-zoom:ccol+zoom],
                        cmap='gray', vmin=0, vmax=1,
                        extent=[-zoom, zoom, zoom, -zoom])
axes5[1].set_title('Şekil 5b: Filtre Maskesi (Merkez Bölgesi)\n(Yakınlaştırılmış görünüm)', fontsize=11, fontweight='bold')
axes5[1].set_xlabel('Δu (merkeze göre)')
axes5[1].set_ylabel('Δv (merkeze göre)')
plt.colorbar(im5b, ax=axes5[1], label='H(u,v)')

plt.tight_layout()
plt.savefig('fig5_filter_mask.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig5_filter_mask.png kaydedildi.")


# ─────────────────────────────────────────────
# ADIM 6: TERS FOURIER VE YENİDEN OLUŞTURMA
# ─────────────────────────────────────────────
print("\n--- ADIM 6: Ters Fourier ve Görüntü Yeniden Oluşturma ---")

G = F_shifted * H
G_log = np.log1p(np.abs(G))
G_unshifted = np.fft.ifftshift(G)
cleaned_raw = np.real(np.fft.ifft2(G_unshifted))

# Normalize et [0, 255]
cleaned = np.clip(cleaned_raw, 0, None)
cleaned = (cleaned / cleaned.max() * 255).astype(np.uint8)
cleaned_f = cleaned.astype(float)

print(f"Filtrelenmiş spektrum: [0, {G_log.max():.3f}] (log)")
print(f"Temizlenmiş görüntü  : [{cleaned.min()}, {cleaned.max()}] piksel değer aralığı")

# Kalite metrikleri
psnr_noisy_clean = psnr(noisy, clean_ref)
psnr_cleaned_clean = psnr(cleaned_f, clean_ref)
psnr_improvement = psnr_cleaned_clean - psnr_noisy_clean
ssim_noisy = ssim(noisy, clean_ref)
ssim_cleaned = ssim(cleaned_f, clean_ref)

print(f"\nKalite Metrikleri:")
print(f"  PSNR (gürültülü vs referans) : {psnr_noisy_clean:.2f} dB")
print(f"  PSNR (temizlenmiş vs referans): {psnr_cleaned_clean:.2f} dB")
print(f"  PSNR iyileştirmesi           : +{psnr_improvement:.2f} dB")
print(f"  SSIM (gürültülü vs referans) : {ssim_noisy:.4f}")
print(f"  SSIM (temizlenmiş vs referans): {ssim_cleaned:.4f}")

# Şekil 6: Filtrelenmiş spektrum
fig6, ax6 = plt.subplots(1, 1, figsize=(6, 6))
im6 = ax6.imshow(G_log, cmap='hot')
ax6.set_title('Şekil 6: Filtrelenmiş Logaritmik Spektrum\nlog(1 + |G(u,v)|) = log(1 + |H·F|)', fontsize=11, fontweight='bold')
ax6.set_xlabel('Frekans u')
ax6.set_ylabel('Frekans v')
plt.colorbar(im6, ax=ax6, label='log(1 + |G|)')
plt.tight_layout()
plt.savefig('fig6_filtered_spectrum.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig6_filtered_spectrum.png kaydedildi.")

# Şekil 7: Temizlenmiş görüntü
fig7, ax7 = plt.subplots(1, 1, figsize=(6, 6))
im7 = ax7.imshow(cleaned, cmap='gray', vmin=0, vmax=255)
ax7.set_title('Şekil 7: Temizlenmiş Görüntü\n(Ters Fourier Dönüşümü ile Yeniden Oluşturulmuş)', fontsize=11, fontweight='bold')
ax7.set_xlabel('Sütun (Piksel)')
ax7.set_ylabel('Satır (Piksel)')
plt.colorbar(im7, ax=ax7, label='Piksel Değeri')
plt.tight_layout()
plt.savefig('fig7_cleaned.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig7_cleaned.png kaydedildi.")

# Bonus: Karşılaştırma figürü
fig8, axes8 = plt.subplots(2, 3, figsize=(15, 10))

axes8[0, 0].imshow(noisy, cmap='gray', vmin=0, vmax=255)
axes8[0, 0].set_title('(a) Gürültülü Görüntü', fontweight='bold')
axes8[0, 0].axis('off')

axes8[0, 1].imshow(log_magnitude, cmap='hot')
axes8[0, 1].set_title('(b) Gürültülü FFT Log-Spektrum', fontweight='bold')
axes8[0, 1].axis('off')

axes8[0, 2].imshow(H, cmap='gray', vmin=0, vmax=1)
axes8[0, 2].set_title('(c) Filtre Maskesi H(u,v)', fontweight='bold')
axes8[0, 2].axis('off')

axes8[1, 0].imshow(cleaned, cmap='gray', vmin=0, vmax=255)
axes8[1, 0].set_title(f'(d) Temizlenmiş Görüntü\nPSNR: {psnr_cleaned_clean:.1f} dB', fontweight='bold')
axes8[1, 0].axis('off')

axes8[1, 1].imshow(G_log, cmap='hot')
axes8[1, 1].set_title('(e) Filtrelenmiş FFT Log-Spektrum', fontweight='bold')
axes8[1, 1].axis('off')

axes8[1, 2].imshow(clean_ref, cmap='gray', vmin=0, vmax=255)
axes8[1, 2].set_title('(f) Referans Temiz Görüntü', fontweight='bold')
axes8[1, 2].axis('off')

fig8.suptitle('Frekans Uzayında Periyodik Gürültü Giderimi — Karşılaştırma\n'
              f'Öğrenci: Omar Nuriyev (24011902)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig8_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] fig8_comparison.png kaydedildi.")

# Temizlenmiş görüntüyü kaydet
Image.fromarray(cleaned).save('cleaned_image.png')
print("\n[OK] cleaned_image.png kaydedildi.")

# ─────────────────────────────────────────────
# ÖZET RAPOR
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ÖZET RAPOR")
print("=" * 60)
print(f"Görüntü boyutu           : {cols}x{rows}")
print(f"Tespit edilen tepe sayısı: {len(peaks)}")
print(f"Filtre tipi              : Gaussian Notch (σ={FILTER_SIGMA})")
print(f"PSNR iyileştirmesi       : +{psnr_improvement:.2f} dB")
print(f"SSIM iyileştirmesi       : {ssim_noisy:.4f} → {ssim_cleaned:.4f}")
print("=" * 60)
print("\nTüm şekiller başarıyla kaydedildi!")
print("Rapor oluşturmak için: python3 generate_report.py")
