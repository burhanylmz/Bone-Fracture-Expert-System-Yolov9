import torch
from transformers import CLIPProcessor, CLIPModel

# Global CLIP instances for lazy loading to prevent Disk I/O bottleneck
_clip_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_clip_model = None
_clip_processor = None

def get_clip_model_and_processor():
    global _clip_model, _clip_processor
    if _clip_model is None or _clip_processor is None:
        print("[CLIP] Model ve işlemci ilk kez yükleniyor (Belleğe Alınıyor)...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_clip_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def run_clip_control(image_rgb) -> dict:
    """
    OpenAI CLIP modelini kullanarak görüntünün röntgen mi yoksa
    doğal bir fotoğraf mı (kedi, manzara vb.) olduğunu tahmin eder.
    """
    print("[CLIP] Geçerlilik kontrolü yapılıyor...")
    
    model, processor = get_clip_model_and_processor()

    # Modele soracağımız iki metin alternatifi (Sınıflarımız)
    candidate_labels = ["a medical X-ray image", "a natural photo of an animal, landscape or object"]

    # Görüntüyü ve metinleri CLIP formatına dönüştür
    inputs = processor(text=candidate_labels, images=image_rgb, return_tensors="pt", padding=True)

    # Girdileri modele ait cihaza taşı
    inputs = {k: v.to(_clip_device) for k, v in inputs.items()}

    # Çıkarım (Inference) yap
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Olasılık değerlerine (Softmax) dönüştür
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Sonuçları etiketlerle eşleştir
    results = {candidate_labels[i]: float(probs[i]) for i in range(len(candidate_labels))}
    
    print(f"[CLIP] Analiz Sonucu -> Röntgen: {results['a medical X-ray image']:.4f}")
    return results