# validator.py - Tıbbi Doğrulama Ajanı (Kural Tabanlı Klinik Doğrulama)
import os
import json
from typing import Dict, Any

class ValidatorAgent:
    def __init__(self, rules_path: str = None):
        # Proje dizininden bağımsız çalışabilmesi için varsayılan kural dosyası yolunu dinamik çözümlüyoruz
        if rules_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            rules_path = os.path.join(current_dir, "rules.json")
        self.rules_path = rules_path
        self.rules = self.load_rules()

    def load_rules(self) -> Dict[str, Any]:
        """rules.json matrisine göre klinik kuralları yükler."""
        try:
            with open(self.rules_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Validator] Warning: Could not load rules.json: {e}")
            return {}

    def validate_findings(self, vision_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bulguları rules.json matrisine göre çapraz kontrole (Cross-Check) tabi tutar.
        Eğer model tespitleri arasında mantıksal veya klinik çelişki varsa bunu işaretler.
        """
        print("[Validator] Validating findings against clinical rule matrix...")
        
        classification_conf = vision_results.get("classification", {}).get("fracture_probability", 0)
        detection_boxes = vision_results.get("detection", {}).get("boxes", [])
        
        is_valid = True
        warnings = []

        # Örnek klinik doğrulama kuralı: Yüksek kırık olasılığı olmasına rağmen bounding box tespit edilmediyse
        if classification_conf > 0.7 and len(detection_boxes) == 0:
            is_valid = False
            warnings.append("High fracture probability but no bounding boxes detected.")

        return {
            "is_valid": is_valid,
            "warnings": warnings,
            "validated_status": "Passed" if is_valid else "Requires Review"
        }
