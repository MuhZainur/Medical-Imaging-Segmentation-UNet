# Clinical Radiograph Segmentation and Pulmonary Biometrics

A medical image segmentation pipeline built in PyTorch using a U-Net architecture paired with Contrast-Limited Adaptive Histogram Equalization (CLAHE). The system segments lung fields from chest radiographs with sub-pixel boundary accuracy and extracts quantitative clinical biometrics, including anatomical lung surface area and cardiothoracic ratio (CTR).

> **Regulatory notice:** This software is developed for research benchmarking and clinical decision support. It is not an FDA-cleared diagnostic device and should not replace clinical review by a certified radiologist.

## Clinical Context and Challenges

Chest radiography is the most common diagnostic imaging exam worldwide. Accurate boundary segmentation of the lung fields is a required foundation for quantifying pulmonary conditions, such as pneumonia consolidation, pneumothorax, chronic obstructive pulmonary disease, and cardiomegaly.

In clinical practice, manual contouring of lung fields is labor intensive and subject to 8% to 12% inter-observer variability between radiologists. When deep learning is applied to raw radiographs, three technical challenges emerge:
1. Low-exposure scans often obscure the costophrenic recesses (where fluid pools) and the retrocardiac space behind the heart.
2. Background soft tissue and image borders occupy roughly 75% of the pixel volume, creating strong class imbalance.
3. Training solely on pixel-level Cross-Entropy loss leads to conservative, eroded masks that misplace the pleural boundary.

This system combines CLAHE preprocessing to normalize local contrast with a Compound Dice-BCE loss objective, stabilizing boundary gradient updates even in regions with poor exposure.

## Technical Architecture

1. **Adaptive contrast equalization (CLAHE):** Radiographs are divided into an 8x8 grid of contextual tiles with a clip limit of 2.0 to enhance alveolar and vascular texture without overamplifying high-frequency bone noise.
2. **PyTorch U-Net architecture:** A four-level encoder-decoder with skip connections transfers high-resolution spatial coordinates directly to the expanding decoder, preserving sharp pleural margins.
3. **Compound objective loss:** A combined Binary Cross-Entropy and Dice loss penalizes outlier pixels while maximizing spatial region overlap.
4. **Biometric extraction:**
   * **Pulmonary surface area (cm²):** Calculated from active mask pixels multiplied by radiographic sensor pixel spacing.
   * **Cardiothoracic ratio (CTR):** Measures cardiac shadow width against maximum internal thoracic diameter. Ratios below 0.50 reflect a normal cardiac silhouette; ratios at or above 0.50 indicate potential cardiomegaly for specialist follow-up.

## Evaluation and Validation

The system was evaluated on the NIH NLM Montgomery County Chest X-Ray database:

| Evaluation Metric | Test Patient (`MCUCXR_0001_0`) | Validation Set Mean | Target Standard |
| :--- | :---: | :---: | :---: |
| Dice similarity coefficient (DSC) | 91.72% (0.9172) | 88.66% | >= 85.0% |
| Intersection over Union (IoU) | 84.71% (0.8471) | 79.67% | >= 75.0% |
| Inference latency per scan (CPU) | 70.0 ms | 71.2 ms | < 150 ms |
| Measured pulmonary surface area | 1289.1 cm² | Anatomically consistent | Diagnostic baseline |
| Cardiothoracic ratio (CTR) | 0.440 (Normal) | Normal range (< 0.50) | Clinical baseline |

The model achieves 70.0 ms inference on a standard CPU, suitable for integration into PACS workstations.

## Documentation and Deliverables

* **Technical publication article:** [`MEDIUM_ARTICLE.md`](./MEDIUM_ARTICLE.md) or [`artikel.md`](./artikel.md) covers CLAHE mathematical formulation, compound loss derivation, and morphometric calculation methods.
* **Video walkthrough:** [`video_presentation.mp4`](./video_presentation.mp4) provides a 1080p narrated presentation of the U-Net architecture and clinical segmentation overlays ([English script transcript](./skenario_video_en.md)).
* **Clinical four-panel report:** [`result_clinical_segmentation.png`](./result_clinical_segmentation.png) presents the raw radiograph, CLAHE enhancement, manual ground truth mask, and U-Net prediction overlay.
* **Radiology audit report:** [`radiology_ai_audit_report.json`](./radiology_ai_audit_report.json) records patient scan metadata, Dice and IoU scores, and biometric area measurements.
* **Trained checkpoint:** [`unet.pt`](./unet.pt) stores the trained model weights.

## Running the Code

### 1. Evaluate a patient scan and generate the clinical report
```bash
python demo.py
```
This loads `unet.pt`, processes the sample radiograph, computes Dice and IoU metrics along with CTR, and writes `result_clinical_segmentation.png` and `radiology_ai_audit_report.json`.

### 2. Train the U-Net model from scratch
```bash
python train.py --epochs 8 --batch-size 4
```
