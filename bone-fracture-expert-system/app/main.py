from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="Bone Fracture Expert System API",
    description="FastAPI Backend for Bone Fracture Classification, Detection, Segmentation and Report Generation.",
    version="1.0.0"
)

# CORS Middleware configurations
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
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # TODO: Implement LangGraph Supervisor invocation and image processing flow
        contents = await file.read()
        return AnalysisResponse(
            status="Success",
            message="X-Ray image received and processed successfully.",
            details={
                "filename": file.filename,
                "findings": "Placeholder analysis findings.",
                "fracture_detected": False
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# localden çıkıp fastapı ile hbys sistemlerine baglanmak için altyapı v2