# DeepFillv2 - Emoji Inpainting

Repository ini merupakan fork dari [nipponjo/deepfillv2-pytorch](https://github.com/nipponjo/deepfillv2-pytorch) yang digunakan untuk tugas kelompok mata kuliah Computer Vision. Fork ini hanya mengambil arsitektur model DeepFillGANv2 serta menggunakan `train.py` untuk melakukan training berdasarkan konfigurasi yang telah ditentukan dan `test.py` untuk melakukan inference.

## Struktur Direktori

```
deepfillv2-pytorch/
├── configs/
│   └── emoji/                  # Konfigurasi training untuk emoji inpainting
│       ├── train-emoji-no_ft.yaml
│       ├── train-emoji-places.yaml
│       └── train-emoji-celebqa.yaml
├── notebooks/                  # Notebook untuk training di Kaggle
│   ├── cv_deepfillganv2.ipynb
│   └── cv-deepfillganv2-kaggle.ipynb
├── scripts/                    # Script evaluasi dan visualisasi
│   ├── compute_l1.py
│   ├── compute_psnr.py
│   ├── compute_ssim.py
│   ├── visualize_predictions.py
│   └── visualize_deepfillv2_loss.py
├── pretrained/                 # Model checkpoint hasil training
├── test.py                     # Script inference
├── train.py                    # Script training
└── run_inference.ps1           # PowerShell script untuk inference & evaluasi
```

## Konfigurasi

Konfigurasi training yang digunakan untuk tugas ini terletak di dalam direktori `configs/emoji/`. Terdapat 3 variasi konfigurasi:

- `train-emoji-no_ft.yaml` - Training dari scratch tanpa fine-tuning
- `train-emoji-places.yaml` - Fine-tuning dari model pre-trained Places2
- `train-emoji-celebqa.yaml` - Fine-tuning dari model pre-trained CelebA-HQ

## Training

Training dilakukan di environment Kaggle menggunakan notebook yang tersedia di direktori `notebooks/`. Notebook utama yang digunakan adalah `cv-deepfillganv2-kaggle.ipynb`.

Untuk menjalankan training secara lokal:

```bash
python train.py --config configs/emoji/train-emoji-no_ft.yaml
```

## Inference & Evaluasi

Untuk menjalankan inference sekaligus evaluasi, gunakan PowerShell script `run_inference.ps1`:

```powershell
# Pilih model secara interaktif
.\run_inference.ps1

# Atau spesifikasikan path model secara langsung
.\run_inference.ps1 -ModelPath "pretrained/states_emoji_no-ft-50000.pth"

# Atau gunakan index model
.\run_inference.ps1 -ModelIndex 0
```

Script ini akan:
1. Menjalankan inference menggunakan `test.py` pada semua gambar di direktori test
2. Menghitung metrik evaluasi (L1, PSNR, SSIM) menggunakan script di `scripts/`
3. Menyimpan hasil prediksi dan metrik evaluasi

### Metrik Evaluasi

Script evaluasi yang tersedia:
- `scripts/compute_l1.py` - Menghitung L1 loss (Mean Absolute Error)
- `scripts/compute_psnr.py` - Menghitung Peak Signal-to-Noise Ratio
- `scripts/compute_ssim.py` - Menghitung Structural Similarity Index

Setiap metrik dihitung untuk:
- **Full Image** - Seluruh area gambar
- **Masked Region** - Hanya area yang di-inpaint

## Visualisasi

### Visualisasi Prediksi

Untuk membuat perbandingan side-by-side antara gambar original, erased, dan hasil inpainting:

```bash
python scripts/visualize_predictions.py \
    --test-dir examples/emoji/triplets_OTI \
    --pred-dir predictions_states_emoji_no-ft-50000 \
    --output comparison.png
```

Script ini berfungsi untuk kedua OTI dan HTI.

## Referensi

- Paper: [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589)
- Original Repository: [nipponjo/deepfillv2-pytorch](https://github.com/nipponjo/deepfillv2-pytorch)

## Contributors

- Vincent Suhardi - 2206082505
- Fadhil Muhammad - 2206083464
- Venedict Chen - 2206024436
- Edbert Halim - 2206813795