import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from app.models.attention_yolo import HybridAttentionAgent, generate_explainable_heatmap

def run_vision_agent_pipeline(state: dict) -> dict:
    if state.get("status") == "invalid_input" or not state.get("is_valid_xray", False):
        return state

    print("\n--- [Adım 3] Röntgen Analiz Ajanı (Vision Agent) Devreye Girdi ---")
    
    # Ön işleme ajanının ürettiği morfolojisi korunmuş dolgulu matrisi alıyoruz
    img_bone_focused = state.get("img_640").copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. MAKALE KATMANI: Hibrit Dikkat Süzgeci Mantığı Entegrasyonu
    img_tensor = torch.from_numpy(img_bone_focused).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    attention_module = HybridAttentionAgent(c_in=3).to(device)
    attention_module.eval()
    
    with torch.no_grad():
        attention_segmented_tensor = attention_module(img_tensor)
        print("🟢 [Hybrid-Attention] İzole edilmiş kemik pikselleri tarandı ve maskelendi.")
        
        attn_img = attention_segmented_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        attn_img = np.clip(attn_img, 0, 255).astype(np.uint8)
        img_input_for_yolo = cv2.resize(attn_img, (640, 640))

    # 2. YOLOv9 LOKALİZASYON KATMANI
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root_dir = os.path.abspath(os.path.join(current_file_dir, "../..")) 
    model_path = os.path.join(project_root_dir, "weights", "best.pt")
    
    if not os.path.exists(model_path):
        print(f"🛑 [HATA] Model bulunamadı: {model_path}")
        return state

    yolo_model = YOLO(model_path).to(device)
    results = yolo_model(img_input_for_yolo, conf=0.10, iou=0.45, imgsz=640, verbose=False)[0]

    detections = []
    is_fracture_present = False
    ensemble_confidence = 0.0
    
    # Varsayılan bölge adını üstteki Anatomi ajanından miras alıyoruz, ezme riskini önlüyoruz
    detected_region_name = state.get("detected_region", "Belirlenemeyen Kemik Bölgesi")

    medical_translation = {
        'elbow positive': 'Dirsek Bölgesi (Fraktür Hatlı)',
        'fingers positive': 'El Parmak Kemikleri (Fraktür Hatlı)',
        'forearm fracture': 'Ön Kol (Radius/Ulna) Kemikleri (Fraktür Hatlı)',
        'humerus fracture': 'Humerus (Üst Kol) Kemiği (Fraktür Hatlı)',
        'humerus': 'Humerus Kemiği (Normal Yapı)',
        'shoulder fracture': 'Omuz Eklemi / Klavikula (Fraktür Hatlı)',
        'wrist positive': 'El Bileği / Karpal Kemikler (Fraktür Hatlı)'
    }
    
    if len(results.boxes) > 0:
        is_fracture_present = True
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            
            raw_class_name = yolo_model.names[class_id]
            tr_class_name = medical_translation.get(raw_class_name, raw_class_name)
            
            # Model dinamik olarak hangi sınıfı bulduysa bölge adı o olur
            detected_region_name = tr_class_name
            
            xai_img_path = generate_explainable_heatmap(img_input_for_yolo, coords)
            detections.append({
                "bone": tr_class_name,
                "fracture_geometry": "Saptanan Süreklilik ve Kortikal İntegre Kaybı",
                "bounding_box": [round(float(c), 2) for c in coords],
                "confidence": round(conf, 4),
                "doctor_visual_report": xai_img_path
            })
        ensemble_confidence = float(results.boxes[0].conf[0].cpu().numpy())
    else:
        is_fracture_present = False
        ensemble_confidence = 0.0
        print("🟢 [YOLOv9] Gerçek Çıkarım: Kemik yapısı temiz, herhangi bir kırık odağı saptanmadı.")

    state["is_fracture_present"] = is_fracture_present
    state["detections"] = detections
    state["ensemble_confidence"] = ensemble_confidence
    state["detected_region"] = detected_region_name  
    state["img_640"] = img_input_for_yolo 
    
    print("--- [Adım 3] Röntgen Analiz Ajanı Görevini Başarıyla Tamamladı ---\n")
    return state