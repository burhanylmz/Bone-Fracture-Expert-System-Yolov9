# 🏥🦴 Multi-Agent Bone Fracture Detection & Clinical Decision Support System (CDSS)

Bu proje, ortopedik travma vakalarında acil servis hekimlerine gerçek zamanlı otonom karar desteği sunmak amacıyla geliştirilmiş, çoklu ajan mimarisine sahip kurumsal bir **Klinik Karar Destek Sistemi (CDSS)** platformudur. 

Sistem; röntgen görüntülerinin semantik doğrulaması, gürültü azaltma, anatomik bölge sınıflandırması, kırık lokalizasyonu ve otonom metin jenerasyonu ile hekimler için anında yapılandırılmış radyoloji raporu üretilmesini sağlar.

---

## 📂 Proje Dizin Yapısı ve Rol Dağılımı

Gereksiz yerel deneme dosyalarının (`src_training`) ve sanal ortamın (`venv`) ana codebase'i kirletmemesi amacıyla profesyonel bir `.gitignore` izolasyonu uygulanmıştır. Çekirdek projenin mimari yapısı şu şekildedir:

*   📂 **`bone-fracture-expert-system/`** — Projenin ana kaynak kod dizini.
    *   📂 **`app/`**
        *   📁 **`agents/`** — İş birliği içinde çalışan LangGraph ajan düğümleri.
            *   📄 `preprocessing_agent.py` — CLAHE kontrast eşitleme ve median blur filtreleme.
            *   📄 `security_agent.py` — CLIP tabanlı girdi doğrulama kontrolü.
            *   📄 `anatomy_agent.py` — EfficientNet-B0 tabanlı anatomik bölge sınıflandırıcısı.
            *   📄 `vision_agent.py` — YOLOv9c ve Hibrit Dikkat (CBAM) süzgeçli teşhis motoru.
            *   📄 `report_agent.py` — Ollama (Llama3) entegrasyonlu otonom raporlayıcı.
            *   📄 `graph.py` — LangGraph akış diyagramı ve durum (State) yönetimi.
        *   📁 **`models/`** — Derin öğrenme mimarileri ve XAI görselleştirme modülleri.
            *   📄 `clip_wrapper.py` — Sıfır örnekli (Zero-Shot) görsel sınıflandırma wrapper'ı.
            *   📄 `attention_yolo.py` — CBAM (Channel/Spatial Attention) ve XAI Isı Haritası üretimi.
        *   📁 **`utils/`** — İş kuralları ve validatör yapıları.
        *   📄 `main.py` — Arka plan API katmanı.
    *   📄 `app_interface.py` — Streamlit tabanlı interaktif hekim paneli.
    *   📄 `requirements.txt` — Bağımlılık paketleri.
*   📂 `src_training/` — *[Git Dışı]* Araştırma, deneme ve model ağırlık eğitim scriptleri.
*   📂 `venv/` — *[Git Dışı]* Lokal Python sanal ortamı.

---

## 🧠 LangGraph Çoklu Ajan (Multi-Agent) Ağ Mimarisi

Sistem, bir hekimin röntgen yüklemesinden raporun teslim edilmesine kadar olan tüm süreci güvenli bir durum grafiği (StateGraph) üzerinden asenkron yönetir:

```text
  [GİRİŞ: Röntgen Yükleme]
             │
             ▼
   ┌───────────────────┐
   │   Preprocessing   │  <─── (CLAHE, Median Blur, Padding)
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │     Security      │  <─── (CLIP ile Röntgen Doğrulama Bariyeri)
   └─────────┬─────────┘
             │
      [Koşullu Geçiş] (check_security_gate)
      /               \
(Röntgen)        (Geçersiz Girdi)
       /                   \
      ▼                     ▼
┌───────────┐         ┌───────────┐
│  Anatomy  │         │  Report   │ <─── (Doğrudan Erken İptal Raporu)
└─────┬─────┘         └─────▲─────┘
      │                     │
      ▼                     │
┌───────────┐               │
│  Vision   │───────────────┘
└───────────┘
```



### 🛡️ 1. Güvenlik ve Giriş Doğrulama Ajanı (Security Agent)
Sisteme yüklenen görselin tıbbi bir X-Ray filmi mi yoksa geçersiz/sabote edici bir doğal fotoğraf mı (manzara, kedi vb.) olduğunu **OpenAI CLIP (ViT-Base-Patch32)** modeliyle doğrular. Röntgen olasılığı **%75**'in altındaysa, akış teşhis motoruna uğramadan doğrudan erken durdurma (Early Stopping) ile bloke edilir.

### 🛠️ 2. Ön İşleme Ajanı (Preprocessing Agent)
Radyolojik görüntünün morfolojisini ezmeden kenarlara dolgu (padding) uygulayarak modeli besler. Görüntüdeki aşırı patlamaları önlemek amacıyla **CLAHE** (`clipLimit=1.5`) kontrast dengeleme ve **Median Blur** gürültü filtreleme uygular.

### 🔬 3. Anatomi Sınıflandırıcı Ajan (Anatomy Agent)
Yüklenen röntgenin hangi kemik bölgesine ait olduğunu (El, Bacak, Kalça, Omuz) **EfficientNet-B0** tabanlı bir sınıflandırıcı ile tespit eder.

### 🚀 4. Röntgen Teşhis ve Vizyon Ajanı (Vision Agent)
*   **Hibrit Dikkat Süzgeci:** Kemik piksellerine odaklanarak arka plan gürültülerini baskılayan **Kanal ve Uzamsal Hibrit Dikkat Mekanizması (CBAM)** devrededir.
*   **Lokalizasyon:** İnce ayar yapılmış **YOLOv9c** mimarisi ile kırık hatlarını pikselsel bazda sınır kutuları (bounding boxes) ile tespit eder.
*   **XAI Akıllı Odak (Smart Zoom):** Saptanan kırık alanını otonom olarak kırpıp, üzerine dikkati temsil eden Grad-CAM ısı haritasını (Heatmap) basarak diske kaydeder.

### 🤖 5. LLM Radyoloji Rapor Ajanı (Report Agent)
Saptanan tüm bulguları ve güven skorlarını yerel **Llama3 (Ollama)** modeline paslayarak hastaya özel, kurumsal ve başhekim ciddiyetinde Türkçe bir klinik rapor üretir.

---

## 💻 Kurulum ve Çalıştırma

### 1. Gereksinimlerin Kurulması
```bash
pip install -r requirements.txt

```

### 2. Yerel LLM Sunucusunun Başlatılması
Sistem yerel olarak Ollama üzerinden Llama3 kullanmaktadır. Ollama kurulu ise terminalden modeli çekip ayaklandırın

```bash

ollama run llama3

```

### 3. Arayüzün Başlatılması 
Hekim paneli arayüzünü ayağa kaldırmak için ana dizinden şu komutu çalıştırın:

```bash

streamlit run bone-fracture-expert-system/app_interface.py

```

Bu proje, Sisoft Ar-Ge Yazılım Mühendisliği stajı kapsamında Burhan YILMAZ tarafından 2026 yılında geliştirilmiştir.
