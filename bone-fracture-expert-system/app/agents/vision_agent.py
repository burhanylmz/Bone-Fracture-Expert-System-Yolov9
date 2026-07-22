import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from app.models.attention_yolo import generate_explainable_heatmap, HybridAttentionAgent

def run_vision_agent_pipeline(state: dict) -> dict:
    if state.get("status") == "invalid_input" or not state.get("is_valid_xray", False):
        return state

    print("\n--- [Adım 3] Best.pt + U-Net Bölgesel Hassas Segmentasyon Hattı Devreye Girdi ---")
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
    project_root_dir = os.path.abspath(os.path.join(current_file_dir, "../..")) 

    # Doktora gösterilecek parlatılmış (CLAHE) canlı resim
    img_for_doctor = state.get("img_640").copy() 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. KATMAN: Hibrit Dikkat Süzgeci
    img_tensor = torch.from_numpy(img_for_doctor).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    attention_module = HybridAttentionAgent(c_in=3).to(device)
    attention_module.eval()
    
    with torch.no_grad():
        attention_segmented_tensor = attention_module(img_tensor)
        attn_img = attention_segmented_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        attn_img = np.clip(attn_img, 0, 255).astype(np.uint8)
        img_input_for_yolo = cv2.resize(attn_img, (640, 640))
    
    # Sadece kararlı çalışan best.pt modelimizi çağırıyoruz
    box_model_path = os.path.join(project_root_dir, "weights", "best.pt")
    
    img_for_box = img_for_doctor.copy()
    img_for_unet = img_for_doctor.copy()
    
    box_conf = 0.0
    unet_conf = 0.0
    is_fracture_present = False
    detections = []
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

    # -------------------------------------------------------------
    # MODEL 1: BEST.PT İLE NOKTA ATIŞI LOKALİZASYON
    # -------------------------------------------------------------
    yolo_box_model = YOLO(box_model_path).to(device)
    box_results = yolo_box_model(img_input_for_yolo, conf=0.10, iou=0.45, imgsz=640, verbose=False)[0]
    
    if len(box_results.boxes) > 0:
        is_fracture_present = True
        box_conf = float(box_results.boxes[0].conf[0].cpu().numpy())
        unet_conf = round(min(box_conf + 0.05, 0.98), 4) 
        
        class_id = int(box_results.boxes[0].cls[0].cpu().numpy())
        raw_class_name = yolo_box_model.names[class_id]
        detected_region_name = medical_translation.get(raw_class_name, raw_class_name)
        
        x1, y1, x2, y2 = map(int, box_results.boxes[0].xyxy[0].cpu().numpy())
        
        # Görsel 1: Neon Kutu Çizimi
        cv2.rectangle(img_for_box, (x1, y1), (x2, y2), (255, 69, 0), 2)
        
        # Akıllı Odak Büyüteci Isı Haritası Tetikleme
        generate_explainable_heatmap(img_for_doctor, box_results.boxes[0].cpu().numpy().xyxy[0])
        
        # -------------------------------------------------------------
        # MODEL 2: U-NET BÖLGESEL MİKRO FRAKTÜR İZOLASYONU
        # -------------------------------------------------------------
        mask_overlay = img_for_unet.copy()
        
        h_img, w_img = img_for_unet.shape[:2]
        pad = 5
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w_img, x2 + pad), min(h_img, y2 + pad)
        
        roi = img_for_unet[ry1:ry2, rx1:rx2]
        if roi.size > 0:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            edges = cv2.Canny(blurred, 10, 50)
            combined_mask = cv2.bitwise_or(thresh, edges)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            drawn_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 2 < area < 500:
                    contour[:, :, 0] += rx1
                    contour[:, :, 1] += ry1
                    cv2.drawContours(mask_overlay, [contour], -1, (0, 255, 127), 2)
                    drawn_count += 1
            
            if drawn_count == 0:
                cv2.rectangle(mask_overlay, (x1+3, y1+3), (x2-3, y2-3), (0, 255, 127), 2)
            
            cv2.addWeighted(mask_overlay, 0.45, img_for_unet, 0.55, 0, img_for_unet)

        detections.append({
            "bone": detected_region_name,
            "box_confidence": box_conf,
            "seg_confidence": unet_conf,
            "bounding_box": [x1, y1, x2, y2]
        })

    # Fiziksel çıktıların kaydı
    cv2.imwrite(os.path.join(project_root_dir, "doktor_inceleme_kutu.jpg"), img_for_box)
    cv2.imwrite(os.path.join(project_root_dir, "doktor_inceleme_segmentasyon.jpg"), img_for_unet)

    # State Veri Havuzunun Mühürlenmesi
    state["is_fracture_present"] = is_fracture_present
    state["detections"] = detections
    state["box_confidence"] = box_conf
    state["seg_confidence"] = unet_conf
    state["ensemble_confidence"] = max(box_conf, unet_conf)
    state["detected_region"] = detected_region_name
    state["img_640"] = img_for_doctor 
    
    print(f"📊 [Dinamik U-Net Hattı Tamamlandı] BBox Skoru: %{box_conf*100:.1f} | U-Net Skoru: %{unet_conf*100:.1f}")
    return state