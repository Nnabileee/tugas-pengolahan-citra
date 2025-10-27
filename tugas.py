import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_float, img_as_ubyte
from skimage.util import random_noise
from skimage.filters import rank
from skimage.morphology import disk
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# === 1. Membaca foto pribadi ===
image_path = "Nabil.jpeg"
image_bgr = cv2.imread(image_path)
if image_bgr is None:
    raise FileNotFoundError(f"Gambar '{image_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")

# Konversi ke grayscale agar sesuai dengan operasi rank filter
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
image = img_as_float(image_gray)

# === 2. Tambahkan noise (misal salt & pepper) ===
noisy = random_noise(image, mode='s&p', amount=0.1)

# === 3. Filtering dengan mean, min, median, dan max filter ===
noisy_ubyte = img_as_ubyte(noisy)

mean_filtered = rank.mean(noisy_ubyte, footprint=disk(3))
min_filtered = rank.minimum(noisy_ubyte, footprint=disk(3))
median_filtered = rank.median(noisy_ubyte, footprint=disk(3))
max_filtered = rank.maximum(noisy_ubyte, footprint=disk(3))

# Kembalikan ke float [0,1] untuk evaluasi
mean_filtered_float = img_as_float(mean_filtered)
min_filtered_float = img_as_float(min_filtered)
median_filtered_float = img_as_float(median_filtered)
max_filtered_float = img_as_float(max_filtered)

# === 4. Evaluasi dengan PSNR dan SSIM ===
metrics = {
    "Noisy": {
        "PSNR": psnr(image, noisy),
        "SSIM": ssim(image, noisy, data_range=1.0)
    },
    "Mean Filtered": {
        "PSNR": psnr(image, mean_filtered_float),
        "SSIM": ssim(image, mean_filtered_float, data_range=1.0)
    },
    "Min Filtered": {
        "PSNR": psnr(image, min_filtered_float),
        "SSIM": ssim(image, min_filtered_float, data_range=1.0)
    },
    "Median Filtered": {
        "PSNR": psnr(image, median_filtered_float),
        "SSIM": ssim(image, median_filtered_float, data_range=1.0)
    },
    "Max Filtered": {
        "PSNR": psnr(image, max_filtered_float),
        "SSIM": ssim(image, max_filtered_float, data_range=1.0)
    }
}

# === 5. Visualisasi hasil ===
fig, axes = plt.subplots(1, 6, figsize=(18, 6))
ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(noisy, cmap='gray')
ax[1].set_title("Noisy")
ax[1].axis('off')

ax[2].imshow(mean_filtered, cmap='gray')
ax[2].set_title("Mean Filter")
ax[2].axis('off')

ax[3].imshow(min_filtered, cmap='gray')
ax[3].set_title("Min Filter")
ax[3].axis('off')

ax[4].imshow(median_filtered, cmap='gray')
ax[4].set_title("Median Filter")
ax[4].axis('off')

ax[5].imshow(max_filtered, cmap='gray')
ax[5].set_title("Max Filter")
ax[5].axis('off')

plt.tight_layout()
plt.show()

# === 6. Cetak hasil evaluasi di terminal ===
print("HASIL EVALUASI FILTERING")
for name, vals in metrics.items():
    print(f"{name:<15} -> PSNR: {vals['PSNR']:.2f}, SSIM: {vals['SSIM']:.4f}")
