import os
import sys
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# MUTLAK DİZİN BAĞLANTISI (Ajan ağını ve modelleri sorunsuz çağırmak için)
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/ klasörü
agents_dir = os.path.join(current_file_dir, "agents")
project_root_dir = os.path.abspath(os.path.join(current_file_dir, "..")) # Kök dizin

if agents_dir not in sys.path: sys.path.insert(0, agents_dir)
if current_file_dir not in sys.path: sys.path.insert(0, current_file_dir)
if project_root_dir not in sys.path: sys.path.insert(0, project_root_dir)

from graph import compiled_graph

app = FastAPI(
    title="Bone Fracture Expert System API",
    description="FastAPI Backend for Bone Fracture Classification, Detection, Segmentation and Report Generation.",
    version="1.0.0"
)

# CORS Middleware configurations (HBYS ve Dış Sistem Entegrasyonu Altyapısı v2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    status: str
    message: str
    details: dict

@app.get("/")
async def root():
    return {
        "app": "Bone Fracture Expert System API",
        "status": "Running",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)):
    """
    Upload an X-ray image for analysis (Classification, Detection, Segmentation, Validation & Report Generation)
    HBYS / PACS Entegrasyonu için Asenkron Servis Katmanı
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    # Kök dizinde geçici önbellek görseli oluşturma
    temp_img_path = os.path.join(project_root_dir, "temp_test_xray.jpg")
    
    try:
        # Görseli diske kaydetme
        with open(temp_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 🤖 LangGraph Çoklu Ajan Ağının Tetiklenmesi
        initial_state = {
            "image_path": temp_img_path,
            "status": "processing",
            "current_agent": "preprocessing"
        }
        
        final_state = compiled_graph.invoke(initial_state)
        
        # NumPy/Image matrislerini JSON çıktısından temizleme
        serializable_state = {}
        for key, value in final_state.items():
            if key not in ["img_orig", "img_640", "img_224"]:
                serializable_state[key] = value
                
        return AnalysisResponse(
            status="Success",
            message="X-Ray image analyzed successfully via LangGraph Multi-Agent Engine.",
            details=serializable_state
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)