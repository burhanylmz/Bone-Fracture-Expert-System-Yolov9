import os
import sys
from typing import TypedDict, Any, List, Dict

#  DİZİN UYUMLULUĞU: Ajan dosyalarının birbirini bulabilmesi için dizin yollarını ekliyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langgraph.graph import StateGraph, END

# Ajanların pipeline fonksiyonlarını içe aktarıyoruz
from security_agent import run_security_agent_pipeline
from preprocessing_agent import run_preprocessing_agent_pipeline
from anatomy_agent import run_anatomy_agent_pipeline
from vision_agent import run_vision_agent_pipeline
from report_agent import run_report_agent_pipeline

#  LANGGRAPH STATE (DURUM) TANIMI
class AgentState(TypedDict, total=False):
    image_path: str
    img_orig: Any
    img_640: Any
    img_224: Any
    is_valid_xray: bool
    detected_region: str
    detections: List[Dict[str, Any]]
    box_confidence: float     #  MİMARİ ENTEGRASYON: Model 1'in ham skoru resmi şemaya bağlandı
    seg_confidence: float     #  MİMARİ ENTEGRASYON: Model 2'nin ham skoru resmi şemaya bağlandı
    ensemble_confidence: float
    is_fracture_present: bool
    preprocessing_status: str
    is_preprocessed: bool
    status: str
    error_message: str
    clip_results: Dict[str, float]
    medical_report_html: str

#  KOŞULLU YÖNLENDİRME FONKSİYONU
def check_security_gate(state: AgentState) -> str:
    """
    [MİMARİ KORUMA BARİYERİ]
    Görsel medikal röntgen olarak doğrulanamadıysa, teşhis ajanlarını atlayarak
    akışı doğrudan raporlama ajanına aktarır (Early Stopping).
    """
    if state.get("status") == "invalid_input" or not state.get("is_valid_xray", False):
        print("🛑 [Graph Router] GÜVENLİK BARİYERİ TETİKLENDİ: Görsel röntgen değil! Teşhis hattı bloke ediliyor, doğrudan rapora geçiliyor.")
        return "report"
    
    print("🟢 [Graph Router] Röntgen başarıyla doğrulandı. Anatomi analiz hattına devam ediliyor.")
    return "anatomy"

#  STATE GRAPH OLUŞTURULMASI
workflow = StateGraph(AgentState)

# 1. Düğümlerin (Nodes) Eklenmesi
workflow.add_node("preprocessing", run_preprocessing_agent_pipeline)
workflow.add_node("security", run_security_agent_pipeline)
workflow.add_node("anatomy", run_anatomy_agent_pipeline)
workflow.add_node("vision", run_vision_agent_pipeline)
workflow.add_node("report", run_report_agent_pipeline)

# 2. Giriş Noktasının Belirlenmesi
workflow.set_entry_point("preprocessing")

# 3. Standart Geçiş
workflow.add_edge("preprocessing", "security")

# 4. Koşullu Geçişin Eklenmesi (Security -> Anatomy VEYA Doğrudan Report)
workflow.add_conditional_edges(
    "security",
    check_security_gate,
    {
        "anatomy": "anatomy",
        "report": "report"
    }
)
# 5. Teşhis ve Raporlama Akış Tamamlanması
workflow.add_edge("anatomy", "vision")
workflow.add_edge("vision", "report")
workflow.add_edge("report", END)

#  GRAFİĞİN DERLENMESİ
compiled_graph = workflow.compile()

if __name__ == "__main__":
    print("✨ LangGraph İş Akışı Mimarisi başarıyla derlendi ve kullanıma hazır!")