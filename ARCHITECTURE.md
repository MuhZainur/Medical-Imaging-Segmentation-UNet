# Arsitektur

```mermaid
flowchart LR
    subgraph Train ["train.py"]
        A[images/ + masks/ PNG] --> B["SegDataset<br/>resize 256, flip, brightness"]
        B --> C["U-Net<br/>encoder MobileNetV2"]
        C --> D["Loss = Dice + BCE"]
        D --> E[Adam]
        C --> F["Val Dice / IoU<br/>simpan bobot terbaik"]
    end
    F --> W[(unet.pt)]
    subgraph Demo ["demo.py"]
        X[1 rontgen] --> Y[U-Net + sigmoid > 0.5]
        W --> Y
        Y --> Z["Plot 3 panel:<br/>asli | ground truth | prediksi"]
    end
```

| File | Peran |
|---|---|
| `src/dataset.py` | pasangan gambar/mask + augmentasi |
| `src/unet_model.py` | pembuat model, metrik Dice/IoU |
| `train.py` | training dan validasi |
| `demo.py` | inference + plot |
