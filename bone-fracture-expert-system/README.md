# Bone Fracture Expert System

Bu proje, röntgen görüntülerinden kemik kırıklarını tespit etmek, segmentasyon yapmak ve tıbbi raporlar oluşturmak amacıyla geliştirilmiş LangGraph tabanlı bir uzman sistemdir.

## Proje Yapısı

```text
bone-fracture-expert-system/
│
├── app/                        # FastAPI Backend Klasörü
│   ├── __init__.py
│   ├── main.py                 # FastAPI API endpoint'leri ve tetikleyiciler
│   │
│   ├── agents/                 # LangGraph Ajanları
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Yönetici Ajan (CLIP entegrasyonlu)
│   │   ├── vision_agent.py     # Röntgen Analiz Ajanı (Modelleri çalıştıran ajan)
│   │   ├── validator.py        # Tıbbi Doğrulama Ajanı
│   │   └── reporter.py         # Tıbbi Raporlama Ajanı (LLM entegrasyonlu)
│   │
│   ├── models/                 # Yapay Zeka Model Yükleyicileri & Weights
│   │   ├── __init__.py
│   │   ├── clip_wrapper.py     # CLIP modelini çağıran fonksiyonlar
│   │   ├── ensemble_models.py  # ResNet50, DenseNet121, EfficientNet ortak çıkarım kodu
│   │   ├── yolo_detector.py    # YOLOv8 .pt dosyasını yükleyen ve koşturan kod
│   │   └── unet_segmenter.py   # nn-UNet segmentasyon çıkarım kodu
│   │
│   └── utils/                  # Yardımcı Fonksiyonlar
│       ├── __init__.py
│       ├── pre_processing.py   # OpenCV CLAHE ve Resizing operasyonları
│       └── rules.json          # Tıbbi Doğrulama Ajanı'nın kural matrisi
│
├── weights/                    # İndirilecek .pt, .pth ağırlık dosyaları (Git'e yüklenmez)
├── requirements.txt            # Proje bağımlılıkları (FastAPI, torch, ultralytics, langgraph)
└── .gitignore                  # Git dışı bırakılacak dosyalar
```

## Kurulum ve Çalıştırma

1. Sanal ortam oluşturun ve aktif edin:
   ```bash
   python -m venv venv
   # Windows için:
   .\venv\Scripts\activate
   # Linux/macOS için:
   source venv/bin/activate
   ```

2. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Modellerin ağırlık dosyalarını (`.pt`, `.pth`) `weights/` klasörüne yerleştirin.

4. FastAPI sunucusunu başlatın:
   ```bash
   uvicorn app.main:app --reload
   ```
