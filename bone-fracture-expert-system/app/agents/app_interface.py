import streamlit as str_web
import os
import sys
import cv2
import numpy as np
from PIL import Image

# MUTLAK DİZİN KİLİTLEME
current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root_dir = os.path.abspath(os.path.join(current_file_dir, "../..")) 

if current_file_dir not in sys.path: sys.path.insert(0, current_file_dir)
if project_root_dir not in sys.path: sys.path.insert(0, project_root_dir)

from graph import compiled_graph

# Sayfa Konfigürasyonu
str_web.set_page_config(page_title="Sisoft Yapay Zeka Kemik Asistanı", layout="wide")

# Gelişmiş CSS Entegrasyonu
str_web.markdown("""
    <style>
        /* Ana ekran arka planı ve yumuşatma */
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            max-width: 95% !important;
        }
        /* Butonları kurumsallaştırma */
        .stButton>button {
            width: 100% !important;
            background: linear-gradient(135deg, #FFF, #E8EAF6) !important;
            color: #1A237E !important;
            font-weight: 700 !important;
            border-radius: 8px !important;
            border: None !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(255,255,255,0.4) !important;
            background: linear-gradient(135deg, #E8EAF6, #FFF) !important;
        }
        /* Expander ve Kart yapılarını modernleştirme */
        .streamlit-expanderHeader {
            background-color: #F8FAFC !important;
            border-radius: 8px !important;
            border: 1px solid #E2E8F0 !important;
            font-weight: 600 !important;
        }
        
        /* 🎯 SOL PANEL BÜTÜNLÜK SİHRETTİ (Sarı okla gösterilen tonla mühürlendi) */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #5C6BC0 0%, #3F51B5 50%, #283593 100%) !important;
            border-right: none !important;
        }
        /* Sol paneldeki tüm metinlerin rengini beyaz/açık tonlara zorluyoruz */
        [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] label {
            color: #E8EAF6 !important;
            font-weight: 500 !important;
        }
        /* Dosya yükleme kutusunun sırıtan beyaz arka planını panele yediriyoruz */
        [data-testid="stSidebarContent"] [data-testid="stFileUploader"] > div {
            background-color: rgba(255, 255, 255, 0.15) !important;
            border: 2px dashed rgba(255, 255, 255, 0.3) !important;
            border-radius: 10px !important;
            color: white !important;
        }
        /* Dosya yükleme butonunun içi */
        [data-testid="stSidebarContent"] [data-testid="stFileUploader"] button {
            background-color: white !important;
            color: #3F51B5 !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

#  Üst Banner & Sisoft Yazılı Logo Alanı
str_web.markdown("""
    <div style='background-color: white; padding: 25px 20px 20px 20px; border-radius: 12px; border: 1px solid #E2E8F0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 25px; text-align: center;'>
        <div style='margin-bottom: 15px;'>
            <img src='https://sisoft.com.tr/tr/imgsirketLogosu/logosizeb/sisoft-s.png' style='height: 75px; object-fit: contain;'>
        </div>
        <h1 style='margin: 0; color: #1A237E; font-family: "Segoe UI", sans-serif; font-weight: 800; font-size: 34px; letter-spacing: -0.5px;'>
            SISOFT YAPAY ZEKA DESTEKLİ KLİNİK KARAR SİSTEMİ
        </h1>
        <p style='margin: 6px 0 0 0; font-size: 15px; color: #64748B; font-family: "Segoe UI", sans-serif; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>
             RADYOLOJİK KEMİK KIRIĞI ANALİZİ VE OTONOM MULTI-AGENT RAPORLAMA PLATFORMU
        </p>
    </div>
""", unsafe_allow_html=True)

#  Sol Panel Entegre Başlığı 
str_web.sidebar.markdown("""
    <div style='padding: 10px 0; text-align: center; margin-top: 10px;'>
        <h3 style='color: white; margin: 0; font-family: "Segoe UI", sans-serif; font-size: 17px; letter-spacing: 0.75px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            📁 RÖNTGEN YÜKLEME MERKEZİ
        </h3>
        <div style='width: 50px; height: 3px; background-color: #FFF; margin: 8px auto 0 auto; border-radius: 2px; opacity: 0.7;'></div>
    </div>
    <br>
""", unsafe_allow_html=True)

# Dosya yükleyici alanı 
uploaded_file = str_web.sidebar.file_uploader("Sisteme işlenecek X-Ray filmini tanımlayın:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_img_path = os.path.join(project_root_dir, "temp_test_xray.jpg")
    with open(temp_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    str_web.sidebar.markdown("<div style='background-color: rgba(200, 230, 201, 0.9); color: #1B5E20; padding: 10px; border-radius: 6px; text-align:center; font-size:13px; font-weight:600; margin-top: 15px; margin-bottom:15px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>✓ Röntgen başarıyla ön belleğe alındı!</div>", unsafe_allow_html=True)
    
    if str_web.sidebar.button("Otonom Analizi Başlat"):
        with str_web.spinner('LangGraph Çoklu Ajan Ağ Mimarisi Tetiklendi...'):
            initial_state = {
                "image_path": temp_img_path,
                "status": "processing",
                "current_agent": "preprocessing"
            }
            
            state = compiled_graph.invoke(initial_state)
            is_valid_xray = state.get("is_valid_xray", False)
            
        with str_web.expander("🛠️ Sistem Mühendisliği Çıktı Logları (State Data)"):
            str_web.write("Durum Değişkenleri (State):", {k: (v if k not in ["img_orig", "img_640", "img_224"] else f"Array{v.shape}") for k, v in state.items()})
            
        if not is_valid_xray:
            str_web.error("🚨 GÜVENLİK İHLALİ: Yüklenen görsel tıbbi bir röntgen filmi (X-Ray) olarak doğrulanamadı!")
            
            if "medical_report_html" in state:
                str_web.html(state["medical_report_html"])
                
            clip_res = state.get("clip_results", {})
            if clip_res:
                xray_p = clip_res.get("a medical X-ray image", 0.0) * 100
                nat_p = clip_res.get("a natural photo of an animal, landscape or object", 0.0) * 100
                str_web.info(f"📊 **Güvenlik Ajanı (CLIP) Analiz Skorları:** Röntgen Olasılığı: %{xray_p:.2f} | Doğal Fotoğraf Olasılığı: %{nat_p:.2f}")
            
            str_web.image(temp_img_path, caption="Analizi Reddedilen Uygunsuz Görsel", width=400)
        else:
            col1, col2 = str_web.columns(2)
            
            detections = state.get("detections", [])
            confidence = state.get("ensemble_confidence", 0.0)
            
            with col1:
                str_web.markdown("<h3 style='color: #1A237E; font-size:20px; margin-bottom:15px;'>🔍 Yapay Zeka Teşhis ve Lokalizasyon</h3>", unsafe_allow_html=True)
                
                tab_genel, tab_zoom = str_web.tabs(["👀 Genel Teşhis Görünümü", "🔎 Akıllı Odak Büyüteci (XAI)"])
                
                if "img_640" in state:
                    img_to_draw = state["img_640"].copy()
                else:
                    img_to_draw = cv2.imread(temp_img_path)
                    img_to_draw = cv2.resize(img_to_draw, (640, 640))
                
                if len(detections) > 0 and state.get("is_fracture_present", False):
                    bbox = detections[0]["bounding_box"]
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    
                    cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"KIRIK (Fracture) %{confidence * 100:.1f}"
                    cv2.putText(img_to_draw, label, (x1, max(20, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    
                    output_full_path = os.path.join(project_root_dir, "doktor_inceleme_tam_ekran.jpg")
                    cv2.imwrite(output_full_path, img_to_draw)
                    
                    with tab_genel:
                        str_web.image(output_full_path, width='stretch', 
                                      caption="Ön İşleme Ajanı Tarafından Süzülmüş Kemik Üzerinde Saptanan Kırık Hattı")
                    
                    with tab_zoom:
                        odak_img_path = os.path.join(project_root_dir, "doktor_inceleme_odak.jpg")
                        if os.path.exists(odak_img_path):
                            str_web.image(odak_img_path, width='stretch',
                                          caption="Grad-CAM Dikkat Isı Haritası ve Akıllı Yakınlaştırılmış Odak Alanı")
                        else:
                            str_web.warning("Kırık saptanamadığı veya model hatası olduğu için büyüteç odaklanamadı.")
                else:
                    with tab_genel:
                        str_web.image(img_to_draw, width='stretch', caption="Röntgen Üzerinde Akut Kırık Saptanamadı.")
                    with tab_zoom:
                        str_web.info("Röntgen üzerinde kırık saptanamadığı için yakınlaştırılmış odak oluşturulmadı.")
                    
            with col2:
                region = state.get("detected_region", "Analiz Edilen Kemik Yapısı")
                str_web.markdown(f"<div style='background: linear-gradient(135deg, #EBF8FF, #E1F5FE); padding: 12px 18px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #0288D1; box-shadow:0 1px 2px rgba(0,0,0,0.05);'><b style='color:#01579B;'>🔬 Lokalize Edilen Anatomik Bölge:</b> <code style='background-color:white; padding:2px 6px; border-radius:4px; color:#0288D1; font-weight:700;'>{region}</code></div>", unsafe_allow_html=True)
                
                if "medical_report_html" in state:
                    custom_height_report = f"<div style='height: 560px; overflow-y: auto;'>{state['medical_report_html']}</div>"
                    str_web.html(custom_height_report)
else:
    #  MODERLEŞTİRİLMİŞ KARŞILAMA DASHBOARD'I
    str_web.markdown("""
        <div style='background: linear-gradient(135deg, #F8FAFC, #F1F5F9); padding: 40px; border-radius: 15px; border: 1px solid #E2E8F0; text-align: center; margin-top: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
            <div style='margin-bottom: 20px;'>
                <img src='https://play-lh.googleusercontent.com/FvJoi6JTJ6xmipBDE2FS2YSCr91EUulL7V5NrLmOTPOFdVdza0U8-dlKJzQS5-LEIrPKmB9BQj3kFBvb1hYdw1U' style='height: 55px; width: 55px; object-fit: contain; filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.15));'>
            </div>
            <h3 style='color: #1A237E; font-family: "Segoe UI", sans-serif; font-weight: 700; margin-bottom: 10px;'>
                Klinik Otomasyon Sistemine Hoş Geldiniz
            </h3>
            <p style='color: #64748B; max-width: 600px; margin: 0 auto 25px auto; font-size: 15px; line-height: 1.5;'>
                Bu panel, ortopedik travma vakalarında acil servis hekimlerine otonom karar desteği sunmak amacıyla geliştirilmiştir. Çoklu ajan mimarisi arka planda röntgen doğrulama, filtreleme, anatomik sınıflandırma ve lokalizasyon görevlerini eş zamanlı yürütür.
            </p>
            <div style='display: flex; justify-content: center; gap: 20px; font-size: 13px; font-weight: 600; color: #475569;'>
                <span style='background: white; padding: 8px 16px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>1. Sol Menüden Röntgen Seçin</span>
                <span style='background: white; padding: 8px 16px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>2. Otonom Analizi Başlatın</span>
                <span style='background: white; padding: 8px 16px; border-radius: 20px; border: 1px solid #E2E8F0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'>3. XAI ve LLM Raporunu İnceleyin</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ENTEGRE GELİŞTİRİCİ KÜNYE KATMANI 
str_web.sidebar.markdown("""
    <br><br><br><br><br><br>
    <div style='background: rgba(255, 255, 255, 0.12); padding: 18px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.2); text-align: center; font-family: sans-serif; backdrop-filter: blur(5px); margin-top: auto;'>
        <p style='color: #E8EAF6; font-size: 10px; margin: 0 0 4px 0; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; opacity: 0.8;'>PROJE GELİŞTİRİCİSİ</p>
        <p style='color: #00E676; font-size: 19px; font-weight: 800; margin: 0; letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>BURHAN</p>
        <p style='color: #FFF; font-size: 12px; margin: 4px 0 0 0; font-weight: 500; opacity: 0.9;'>Ar-Ge Yazılım Mühendisliği Stajyeri</p>
        <div style='margin: 12px auto; width: 45px; height: 1.5px; background-color: rgba(255, 255, 255, 0.25);'></div>
        <p style='color: #E8EAF6; font-size: 10px; margin: 0; font-weight: 600; opacity: 0.6;'>Sisoft Sağlık Bilgi Sistemleri © 2026</p>
    </div>
""", unsafe_allow_html=True)