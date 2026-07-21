import os
import sys
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Dosya ağacına göre mutlak yol kitleme (app/agents/graph.py'ye ulaşmak için)
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/api/
app_dir = os.path.abspath(os.path.join(current_file_dir, "..")) # app/
project_root_dir = os.path.abspath(os.path.join(app_dir, "..")) # bone-fracture-expert-system/

agents_dir = os.path.join(app_dir, "agents")
if agents_dir not in sys.path:
    sys.path.insert(0, agents_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from graph import compiled_graph

router = APIRouter(prefix="/api/v1", tags=["Radyoloji Analiz Engine"])

@router.post("/analyze")
async def analyze_xray(file: UploadFile = File(...)):
    """
    Röntgen filmini kabul eder, LangGraph Çoklu Ajan Ağını tetikler 
    ve sonucu JSON payload olarak asenkron döner.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Yüklenen dosya geçerli bir görsel formatı değil.")

    # Temp dosya yönetimi (Proje kök dizinine kaydeder)
    temp_img_path = os.path.join(project_root_dir, "temp_test_xray.jpg")
    
    try:
        with open(temp_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 🤖 LangGraph Çoklu Ajan Ağının Tetiklenmesi
        initial_state = {
            "image_path": temp_img_path,
            "status": "processing",
            "current_agent": "preprocessing"
        }
        
        final_state = compiled_graph.invoke(initial_state)
        
        # Array/Numpy objelerini JSON uyumlu hale getirme
        serializable_state = {}
        for key, value in final_state.items():
            if key not in ["img_orig", "img_640", "img_224"]: # Ağır matrisleri JSON payload'ından çıkarıyoruz
                serializable_state[key] = value
                
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": "LangGraph Ajan Analizi Başarıyla Tamamlandı",
            "data": serializable_state
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ajan Ağı Analiz Hatası: {str(e)}")