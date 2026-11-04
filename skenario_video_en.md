# Video Presentation Scenario (English): Lung Segmentation with U-Net

> **Voice:** Natural Conversational English Narration (ElevenLabs / Azure Neural Speech)
> **Total Duration:** 148.69s (~2.48 min, 4460 frames @ 30 FPS)
> **Format:** 1920x1080 Full HD, Remotion MP4

---

### S01: Lung Segmentation with U-Net
- **Duration:** 10.95s audio + 0.5s pause = 11.45s (343 frames)
- **On-Screen Title:** `Lung Segmentation with U-Net`
- **On-Screen Subtitle:** `Educational research tool, not certified for clinical diagnosis`
- **Narration (Spoken):**
  > "This project explores automated lung boundary delineation on chest radiographs using deep learning. It is an educational engineering prototype, not a certified medical diagnostic device."

### S02: Manual Contouring: Time-Consuming and Subjective
- **Duration:** 18.47s audio + 0.5s pause = 18.97s (569 frames)
- **On-Screen Title:** `Manual Contouring: Time-Consuming and Subjective`
- **On-Screen Subtitle:** `Automated segmentation enables consistent extraction of anatomical surface metrics`
- **Narration (Spoken):**
  > "Manually contouring lung fields on chest radiographs takes significant radiologist time and introduces inter-observer variability. If a computer vision model can produce consistent, repeatable contours, clinicians can derive quantitative anatomical metrics, such as total lung surface area."

### S03: CLAHE Contrast -> 256x256 Resize -> U-Net -> Probability Mask
- **Duration:** 29.49s audio + 0.5s pause = 29.99s (900 frames)
- **On-Screen Title:** `CLAHE Contrast -> 256x256 Resize -> U-Net -> Probability Mask`
- **On-Screen Subtitle:** `Skip connections preserve high-frequency pleural contours (>0.50 threshold)`
- **Narration (Spoken):**
  > "First, radiograph contrast is enhanced using CLAHE to accentuate alveolar and pleural boundaries, then resized to two hundred fifty-six by two hundred fifty-six pixels. The image enters a U-Net architecture. Its contracting path captures contextual representation, the expanding path reconstructs spatial resolution, and skip connections preserve high-frequency edge details. Pixels with predicted lung probability above fifty percent form the final mask."

### S04: Montgomery County Chest X-Ray Set
- **Duration:** 18.42s audio + 0.5s pause = 18.92s (567 frames)
- **On-Screen Title:** `Montgomery County Chest X-Ray Set`
- **On-Screen Subtitle:** `10 image-mask pairs in repository; trained with combined Dice and BCE loss`
- **Narration (Spoken):**
  > "The dataset is from the Montgomery County Chest X-Ray collection, with ten image-mask pairs annotated by radiologists in this project folder. The network was trained using a composite Dice and Binary Cross-Entropy loss, which balances pixel-level accuracy with foreground-background class imbalance."

### S05: Single Test Patient: Dice 0.9172 | IoU 0.8471 | 70.02 ms
- **Duration:** 30.22s audio + 0.5s pause = 30.72s (922 frames)
- **On-Screen Title:** `Single Test Patient: Dice 0.9172 | IoU 0.8471 | 70.02 ms`
- **On-Screen Subtitle:** `Estimated anatomical lung area: 1,289.1 cm² (Assuming 0.175 mm pixel spacing)`
- **Narration (Spoken):**
  > "On a sample test patient, the overlap between the model prediction and the radiologist ground truth achieved a Dice similarity coefficient of zero point nine one seven two and an Intersection over Union of zero point eight four seven one. Inference latency was seventy point zero two milliseconds on CPU. From the predicted mask, anatomical lung surface area was estimated at one thousand two hundred eighty-nine point one square centimeters, assuming a standard pixel spacing of zero point one seven five millimeters."

### S06: Critical Note: Single Patient Case Study
- **Duration:** 23.17s audio + 0.5s pause = 23.67s (710 frames)
- **On-Screen Title:** `Critical Note: Single Patient Case Study`
- **On-Screen Subtitle:** `Train-test separation was not verified per patient; area assumes standard spacing`
- **Narration (Spoken):**
  > "These numbers require careful interpretation. This evaluation represents a single patient case study, and I did not independently verify patient-level separation from the training set, meaning this does not prove generalization to unseen patients. Additionally, the surface area calculation relies on an assumed physical pixel dimension rather than calibrated DICOM metadata."

### S07: Next Steps: Multi-Center Patient-Stratified Cohorts
- **Duration:** 14.47s audio + 0.5s pause = 14.97s (449 frames)
- **On-Screen Title:** `Next Steps: Multi-Center Patient-Stratified Cohorts`
- **On-Screen Subtitle:** `Educational prototype for AI research only, not for clinical diagnosis`
- **Narration (Spoken):**
  > "Future development requires rigorous patient-stratified cross-validation on multi-center hospital cohorts. As a reminder, this prototype is for engineering study and educational purposes only. Thank you for watching."

