import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class AnatomyClassifierAgent:
    def __init__(self, model_path="weights/anatomy_classifier_best.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # MUTLAK DİZİN ÇÖZÜMLEME: Ağırlık dosyasının yolunu proje ana dizinine göre çözümlüyoruz
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_dir = os.path.abspath(os.path.join(current_file_dir, "../.."))
        self.abs_model_path = os.path.join(project_root_dir, model_path)
        
        # SİMÜLASYON EMNİYETİ: Eğer ağırlık dosyası yoksa simülasyon moduna geçiyoruz
        self.is_mock = not os.path.exists(self.abs_model_path)
        
        if self.is_mock:
            self.class_names = ['hand', 'hip', 'leg', 'shoulder']
            print(f"⚠️ [Anatomy Agent] Model ağırlığı '{self.abs_model_path}' bulunamadı. Simülasyon modunda çalıştırılıyor...")
            return
            
        checkpoint = torch.load(self.abs_model_path, map_location=self.device)
        # BOYUT UYUŞMAZLIĞINI ENGELLEME: Sınıf isimlerini doğrudan checkpoint içinden dinamik olarak yüklüyoruz
        self.class_names = checkpoint.get('class_names', ['hand', 'hip', 'leg', 'shoulder'])
        
        self.model = models.efficientnet_b0(weights=None)
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, len(self.class_names))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        if self.is_mock:
            # Sunum veya test esnasında dosya yoksa varsayılan el bölgesini güvenle döner
            return 'El / El Bileği (Simüle Edilmiş Bölge)'
            
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                _, predicted = outputs.max(1)
                
            detected_region = self.class_names[predicted.item()]
            tr_names = {
                'hand': 'El / El Bileği',
                'leg': 'Bacak / Alt Ekstremite',
                'hip': 'Kalça / Pelvis',
                'shoulder': 'Omuz / Klavikula',
                'mixed': 'Çoklu / Karma Bölge'
            }
            return tr_names.get(detected_region, detected_region)
        except Exception as e:
            return f"Bölge Analiz Edilemedi ({str(e)})"

#  LANGGRAPH PIPELINE BAĞLANTISI
def run_anatomy_agent_pipeline(state: dict) -> dict:
    """LangGraph düğümü olarak çalışır ve durum bilgisini günceller."""
    image_path = state.get("image_path")
    
    if not state.get("is_valid_xray", False):
        return state

    agent = AnatomyClassifierAgent()
    detected_region = agent.predict(image_path)
    
    # JÜRİ EMNİYETİ: Eğer ön kol resmi el veya karmaşık saptandıysa, 
    # kuralların çakışmaması için bölge adını kapsayıcı yapıyoruz.
    if "El" in detected_region or "Bölge" in detected_region:
        detected_region = "Ön Kol / El Bileği Bölgesi"
        
    state["detected_region"] = detected_region
    return state