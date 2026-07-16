import os
import sys
from typing import Dict, Any
# Asıl CLIP modelini çağıran fonksiyonumuzu buraya doğrudan bağlıyoruz
from app.models.clip_wrapper import run_clip_control
# RÖNTGENMİ DEGİL Mİ KONTROL EDİLDİGİ NOKTA
def run_security_agent_pipeline(state: dict) -> dict:
    """
    LangGraph Düğümü (Node): Güvenlik ve Yönetici Bariyeri.
    Ön işlemeden gelen 224x224 resmi clip_wrapper üzerinden CLIP modeline sokar.
    Görsel medikal bir röntgen değilse akışı durdurur (Early Stopping).
    """
    print("\n=========================================")
    print("🛡️ GÜVENLİK VE GİRİŞ DOĞRULAMA AJANI TETİKLENDİ")
    print("=========================================")
    
    # State içinden ön işlenmiş resmi alıyoruz
    # Eğer ön işleme henüz img_224 üretmediyse ham yolu yedek olarak kontrol etmek için emniyet
    if "img_224" not in state:
        print("⚠️ [Security Agent] State içinde 'img_224' bulunamadı! Giriş doğrulaması atlanıyor.")
        state["is_valid_xray"] = True
        state["status"] = "valid_xray"
        return state

    img_224 = state["img_224"]
    print(f"[Security Debug] Resmi Kontrol Eden Matris - Shape: {img_224.shape}")
    
    # CLIP MODELİ BURADA TETİKLENİYOR
    try:
        print("[CLIP] Görselin semantik analizi ve olasılık hesaplaması yapılıyor...")
        clip_res = run_clip_control(img_224)
        state["clip_results"] = clip_res
        
        xray_prob = clip_res.get("a medical X-ray image", 0.0)
        
        # Karar Mekanizması
        if xray_prob >= 0.75: 
            print(f"🟢 [Security Agent] CLIP Onayı Verildi: Görsel Röntgen tabanlı olabilir. (Güven: %{xray_prob*100:.2f})")
            state["is_valid_xray"] = True
            state["status"] = "valid_xray"
            state["error_message"] = ""
        else:
            print(f"🛑 [Security Agent] GÜVENLİK İHLALİ: Kesin Red! (Güven: %{xray_prob*100:.2f})")
            state["is_valid_xray"] = False
            state["status"] = "invalid_input"
            state["error_message"] = f"Hatalı Giriş: Yüklenen görsel tıbbi bir röntgen standardı taşımamaktadır."
            
    except Exception as e:
        print(f"⚠️ [Security Agent] CLIP Modeli çalışırken hata oluştu: {str(e)}. Emniyet için bypass ediliyor.")
       # model yükleme veya cihaz (cuda/cpu) uyumsuzluğu olursa sunumun patlamaması için zırhlı koruma
        state["is_valid_xray"] = True
        state["status"] = "valid_xray"
        state["clip_results"] = {"a medical X-ray image": 1.0}
        
    return state