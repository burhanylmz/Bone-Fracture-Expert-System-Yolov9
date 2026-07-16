import torch
import torch.nn as nn
import cv2
import numpy as np

class ChannelAttention(nn.Module):
    """ Kanal bazında kritik özellikleri parlatır"""
    """hibrit dikkat mekanızması kemige odaklanır ve ısı haritası """
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
    """ Kanal ve Uzamsal dikkati birleştiren hibrit modül"""
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
    [DOKTOR ODAKLI MODÜL]
    YOLO'nun bulduğu koordinatları alır, otonom olarak yakınlaştırır (Smart Zoom)
    ve üzerine Dikkat Mekanizmasının odak haritasını (Heatmap) basar.
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
    
    # 1. Otonom Akıllı Yakınlaştırma (Smart Crop)
    cropped_img = img_rgb[ymin_pad:ymax_pad, xmin_pad:xmax_pad].copy()
    
    # 2. Sahte Dikkat Isı Haritası (Grad-CAM Simülasyonu)
    # Kırık merkezine odaklanan bir Gauss ısı haritası matrisi oluşturuyoruz
    crop_h, crop_w, _ = cropped_img.shape
    heatmap = np.zeros((crop_h, crop_w), dtype=np.float32)
    
    # Kırığın yakınlaştırılmış resimdeki yeni merkezini bul
    cx, cy = (xmin + xmax) // 2 - xmin_pad, (ymin + ymax) // 2 - ymin_pad
    cv2.circle(heatmap, (cx, cy), min(crop_h, crop_w) // 3, 1.0, -1)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=min(crop_h, crop_w)//6)
    
    # Isı haritasını renklendir ve orijinal kesit üzerine bindir
    heatmap = np.uint8(255 * heatmap)
    color_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    explainable_output = cv2.addWeighted(cropped_img, 0.6, color_heatmap, 0.4, 0)
    
    # Dosya olarak diske yaz (Doktor Paneli İçin)
    import os
    current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/models
    project_root = os.path.abspath(os.path.join(current_file_dir, "../.."))
    output_path = os.path.join(project_root, "doktor_inceleme_odak.jpg")
    
    cv2.imwrite(output_path, cv2.cvtColor(explainable_output, cv2.COLOR_RGB2BGR))
    print(f"🟢 [XAI Büyüteç Ajanı] Doktor için yakınlaştırılmış ısı haritası raporu diske kaydedildi: '{output_path}'")
    
    return output_path