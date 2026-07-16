import os
import cv2
import numpy as np

def resize_with_padding(img: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    [EN STABİL BOYUTLANDIRMA]
    veri setindeki dikey ve yatay tüm ekstrem röntgenlerin morfolojisini korur.
    Resmi ezmeden, kenarlara siyah dolgu ekleyerek hedefe ulaştırır.
    """
    h, w = img.shape[:2]
    expected_w, expected_h = target_size
    
    scale = min(expected_w / w, expected_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)
    
    dx = (expected_w - new_w) // 2
    dy = (expected_h - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized
    return canvas

def run_preprocessing_agent_pipeline(state: dict) -> dict:
    print("\n--- [Adım 1] Ön İşleme Ajanı (Preprocessing Agent) Devreye Girdi ---")
    
    img_path = state.get("image_path")
    if not img_path or not os.path.exists(img_path):
        state["preprocessing_status"] = "HATA: İşlenecek resim bulunamadı."
        return state

    img = cv2.imread(img_path)
    if img is None:
        state["preprocessing_status"] = "HATA: Resim okunamadı."
        return state

    print(f"🔄 [Preprocessing Agent] Orijinal Resim Boyutu: {img.shape[1]}x{img.shape[0]}")
    
    # Agresif kırpma veri setindeki uzun kemik kırıklarını kadraj dışına fırlattığı için bu aşamada resmin orijinal bütünlüğünü  koruyoruz.

    # 1. Tıbbi Kontrast Dengesi (CLAHE) - Patlamayı önlemek için clipLimit 1.5'e çekildi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_res = clahe.apply(gray)
    
    # 2. Minimum Gürültü Filtresi (Median Blur)
    denoised_res = cv2.medianBlur(clahe_res, 3)
    
    final_img = cv2.cvtColor(denoised_res, cv2.COLOR_GRAY2BGR)
    
    # İşlenmiş kararlı matrisi diske yazıyoruz
    cv2.imwrite(img_path, final_img)
    
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    state["img_orig"] = final_img_rgb
    
    # En-boy oranını koruyarak ajanlar için belleğe alıyoruz
    state["img_640"] = resize_with_padding(final_img_rgb, (640, 640))
    state["img_224"] = resize_with_padding(final_img_rgb, (224, 224))
    
    state["preprocessing_status"] = "BAŞARILI: Stabil CLAHE ve Oran Koruyucu Boyutlandırma uygulandı."
    state["is_preprocessed"] = True
    
    print("🟢 [Preprocessing Agent] FracAtlas veri seti için en stabil ön işleme parametreleri uygulandı.")
    print("--- [Adım 1] Ön İşleme Ajanı Görevini Tamamladı ---")
    return state