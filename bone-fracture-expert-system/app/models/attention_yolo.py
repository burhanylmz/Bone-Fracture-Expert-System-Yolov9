import os
import torch
import torch.nn as nn
import cv2
import numpy as np

class ChannelAttention(nn.Module):
    """Kanal bazında kritik özellikleri parlatır (Hibrit dikkat mekanizması)"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        mid_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Görüntüde kırık olan piksel konumlarına odaklanır"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class HybridAttentionAgent(nn.Module):
    """Kanal ve Uzamsal dikkati birleştiren hibrit modül"""
    def __init__(self, c_in):
        super(HybridAttentionAgent, self).__init__()
        self.ca = ChannelAttention(c_in)
        self.sa = SpatialAttention()

    def forward(self, x):
        # Denklem (7): Önce kanal dikkati uygulanır
        x_out = x * self.ca(x)
        # Denklem (13): Sonra uzamsal dikkat ile rafine edilir
        x_out = x_out * self.sa(x_out)
        return x_out

def generate_explainable_heatmap(img_rgb, bounding_box):
    """
    [DOKTOR ODAKLI ÇİFT ÇIKTI MODÜL]
    1. Tüm resim üzerinde kırık odağına ısı haritası basar.
    2. Sadece kırık odağını ısı haritasız, temiz bir şekilde yakınlaştırır (Smart Zoom).
    """
    h, w, _ = img_rgb.shape
    xmin, ymin, xmax, ymax = map(int, bounding_box)
    
    # %25 anatomik genişletme payı (Padding)
    pad_w = int((xmax - xmin) * 0.25)
    pad_h = int((ymax - ymin) * 0.25)
    
    xmin_pad = max(0, xmin - pad_w)
    ymin_pad = max(0, ymin - pad_h)
    xmax_pad = min(w, xmax + pad_w)
    ymax_pad = min(h, ymax + pad_h)
    
    # -------------------------------------------------------------
    # FOTOĞRAF 1: GENEL RESİM ÜZERİNDE BÖLGESEL ISI HARİTASI
    # -------------------------------------------------------------
    full_heatmap = np.zeros((h, w), dtype=np.float32)
    cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
    # Sadece kırık merkezine dairesel bir odak açıyoruz
    cv2.circle(full_heatmap, (cx, cy), min(xmax-xmin, ymax-ymin) // 2, 1.0, -1)
    full_heatmap = cv2.GaussianBlur(full_heatmap, (0, 0), sigmaX=min(xmax-xmin, ymax-ymin)//3)
    
    full_heatmap = np.uint8(255 * full_heatmap)
    color_heatmap = cv2.applyColorMap(full_heatmap, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Orijinal genel görüntünün üzerine ısı haritasını transparan olarak yediriyoruz
    explainable_full = cv2.addWeighted(img_rgb, 0.7, color_heatmap, 0.3, 0)
    
    # -------------------------------------------------------------
    # FOTOĞRAF 2: ISI HARİTASIZ, SADECE NET YAKINLAŞTIRILMIŞ ODAK (SMART CROP)
    # -------------------------------------------------------------
    cropped_clean = img_rgb[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()
    
    # Dosya yollarını çözümlüyoruz
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, "../.."))
    
    output_path_full = os.path.join(project_root, "doktor_inceleme_odak_full.jpg")
    output_path_crop = os.path.join(project_root, "doktor_inceleme_odak_crop.jpg")
    
    # Diske kayıt (OpenCV BGR bekler)
    cv2.imwrite(output_path_full, cv2.cvtColor(explainable_full, cv2.COLOR_RGB2BGR))
    cv2.imwrite(output_path_crop, cv2.cvtColor(cropped_clean, cv2.COLOR_RGB2BGR))
    
    print(f"🟢 [XAI Büyüteç Ajanı] Genel Isı Haritası: {output_path_full}")
    print(f"🟢 [XAI Büyüteç Ajanı] Temiz Yakınlaştırılmış Odak: {output_path_crop}")
    
    return output_path_full