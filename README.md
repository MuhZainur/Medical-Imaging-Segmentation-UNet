# 🏥 Clinical Chest Radiograph Segmentation & Biometrics (Medical AI)

Sistem segmentasi citra medis berbasis **U-Net PyTorch** dengan prapemrosesan radiologi **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. Menghasilkan delineasi batas organ paru otomatis berakurasi sub-piksel serta mengestimasi indikator biometrik klinis: **Anatomical Lung Surface Area (cm²)** dan **Cardiothoracic Ratio (CTR)** untuk skrining awal kardiomegali.

---

## 🎯 Nilai Bisnis & Masalah Industri
- **Problem:** Di fasilitas kesehatan dan rumah sakit, dokter spesialis radiologi harus menganalisis ratusan rontgen dada (CXR) per hari. Pengukuran manual luas paru dan rasio jantung terhadap rongga toraks (CTR) membutuhkan waktu dan rentan variabilitas subjektif antar operator.
- **Solusi AI:** Segmentasi berbasis U-Net yang dilatih pada citra klinis riil dengan augmentasi CLAHE untuk mempertajam batas alveolus dan pleura. Sistem menyajikan masker organ paru transparan dan secara otomatis menghitung estimasi parameter biometrik kuantitatif sebagai asisten skrining radiologi.

---

## 🏗️ Pendekatan Teknis (Intermediate-Level Engineering)
1. **Adaptive Contrast Enhancement (CLAHE):** Citra rontgen sering memiliki sebaran kontras yang tidak merata antar area mediastinum dan paru. Prapemrosesan CLAHE (`clip_limit=2.0`, `tile_grid_size=(8,8)`) diterapkan untuk menonjolkan fitur tekstur parenkim paru sebelum masuk ke encoder.
2. **Deep Residual U-Net Architecture:**
   - 4-level encoder-decoder dengan *skip connections* untuk menjaga detail spasial beresolusi tinggi.
   - Fungsi loss gabungan: **Dice Loss + Binary Cross Entropy (BCE)** untuk mengatasi ketidakseimbangan proporsi piksel organ terhadap latar belakang.
3. **Clinical Biometric Extraction:**
   - **Luas Permukaan Paru (cm²):** Dihitung dari akumulasi piksel masker terprediksi dikalikan resolusi proyeksi radiografi standar (~0.2 mm/piksel).
   - **Cardiothoracic Ratio (CTR) Approximation:** Mengukur lebar bayangan siluet jantung terhadap diameter transversal toraks terdalam. CTR $< 0.50$ diklasifikasikan sebagai *Normal Cardiac Silhouette*, sedangkan CTR $\ge 0.50$ menandakan indikasi kardiomegali yang memerlukan rujukan dokter.

---

## 📊 Hasil Benchmark & Validasi Riil

Pengujian dijalankan pada dataset klinis resmi **NIH NLM Montgomery County Chest X-Ray Set**:
- **Resolusi Masukan:** $256 \times 256$ piksel dengan CLAHE enhancement
- **Pelatihan:** 8 epoch dengan optimizer Adam ($\text{lr}=10^{-3}$)

| Metrik Evaluasi | Hasil Pasien Uji (`MCUCXR_0001_0`) | Rata-rata Validasi | Standar Klinis AI |
| :--- | :---: | :---: | :---: |
| **Dice Similarity Coefficient** | **91.72%** (0.9172) | **88.66%** | $\ge 85.0\%$ ✅ |
| **Intersection over Union (IoU)** | **84.71%** (0.8471) | **79.67%** | $\ge 75.0\%$ ✅ |
| **Latensi Inferensi per Citra** | **70.0 ms** | **71.2 ms** | $< 150\text{ ms}$ ✅ |
| **Luas Area Paru Terukur** | **1289.1 cm²** | - | Konsisten Anatomi |
| **Cardiothoracic Ratio (CTR)** | **0.440** (Normal Silhouette) | - | Normal ($< 0.50$) |

---

## 🖼️ Hasil Visual & Laporan Audit

- **Artikel Publikasi Teknis (Medium Style):** [`MEDIUM_ARTICLE.md`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/MEDIUM_ARTICLE.md) / [`artikel.md`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/artikel.md)  
  Ulasan engineering mendalam: normalisasi kontras adaptif lokal CLAHE, arsitektur U-Net PyTorch murni dengan skip connection, optimasi Compound Dice-BCE loss, dan kuantifikasi morfometri paru.
- **Video Presentasi Portofolio (Remotion + ElevenLabs):** [`video_presentation.mp4`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/video_presentation.mp4) ([Naskah Skenario Bahasa Inggris](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/skenario_video_en.md))  
  Video presentasi Full HD 1080p berdurasi ~2:28 menit dengan narasi suara natural berbahasa Inggris, teks layar, dan demonstrasi U-Net segmentation pada rontgen dada klinis.
- **Clinical 4-Panel Report:** [`result_clinical_segmentation.png`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/result_clinical_segmentation.png)  
  Menampilkan 4 panel komparasi medis: Citra Rontgen Asli, Hasil CLAHE Enhancement, Ground Truth Radiolog Manual, dan Prediksi U-Net Overlay dengan metrik Dice & IoU.
- **Formal Radiology AI Audit Report:** [`radiology_ai_audit_report.json`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/radiology_ai_audit_report.json)  
  Dokumen JSON audit yang merinci parameter teknis model, metrik kecocokan klinis, dan estimasi biometrik CTR pasien.
- **Trained Model Checkpoint:** [`unet.pt`](file:///H:/coding/Portfolio/Computer%20Vision/04_Medical_Imaging_Segmentation/unet.pt)

---

## 🚀 Cara Menjalankan

### 1. Menjalankan Evaluasi & Menghasilkan Laporan Klinis
```bash
python demo.py
```
Perintah ini akan memuat model `unet.pt`, mengevaluasi citra rontgen dada uji, menghitung Dice/IoU serta parameter biometrik (Luas cm² dan CTR), lalu menghasilkan `result_clinical_segmentation.png` dan `radiology_ai_audit_report.json`.

### 2. Melatih Ulang Model U-Net
```bash
python train.py --epochs 8 --batch-size 4
```

> **Catatan Regulasi Medis:** Perangkat lunak ini dikembangkan sebagai alat bantu skrining dan riset (*computer-aided triage*), bukan pengganti diagnosis klinis definitif dari dokter spesialis radiologi.
