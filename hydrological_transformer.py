import torch
import torch.nn as nn

class FluxTransformer(nn.Module):
    """
    Bu modül, sistemin mevcut durumunu (tüm hazneler ve atmosferik girdiler) alıp,
    hazneler arasındaki su akılarını (fluxes) tahmin eden bir Transformer ağıdır.
    """
    def __init__(self, input_channels, model_dim=64, n_head=4, n_layers=2):
        super().__init__()
        
        # 1. Girdi Gömme (Input Embedding)
        # Farklı girdileri (yağış, sıcaklık, hazne dolulukları vb.) tek bir zengin
        # özellik haritasına dönüştürmek için bir evrişim katmanı kullanırız.
        self.input_embed = nn.Conv2d(input_channels, model_dim, kernel_size=1)
        
        # 2. Transformer Kodlayıcı (Encoder)
        # Bu, modelin "beyni"dir. Gömülmüş özellik haritasındaki uzamsal desenleri
        # ve ilişkileri öğrenir.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=n_head, 
            dim_feedforward=model_dim * 4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Çıktı Katmanları (Output Heads)
        # Transformer'dan gelen zengin özellik haritasını, her biri belirli bir akıyı
        # (sızma, akış vb.) temsil eden ayrı haritalara dönüştürür.
        # Her akı için ayrı bir "kafa" (head) oluşturuyoruz.
        self.head_inf = nn.Conv2d(model_dim, 1, kernel_size=1) # Sızma (Infiltration)
        self.head_aet = nn.Conv2d(model_dim, 1, kernel_size=1) # Gerçek Buharlaşma (AET)
        self.head_cap = nn.Conv2d(model_dim, 1, kernel_size=1) # Kılcal Yükselim (Capillary Rise)
        self.head_per = nn.Conv2d(model_dim, 1, kernel_size=1) # Derine Sızma (Percolation)
        self.head_q1 = nn.Conv2d(model_dim, 1, kernel_size=1)  # Yüzey Akışı (q1)
        self.head_q2 = nn.Conv2d(model_dim, 1, kernel_size=1)  # Yüzey Altı Akışı (q2)
        self.head_q3 = nn.Conv2d(model_dim, 1, kernel_size=1)  # Yeraltı Suyu Akışı (q3)

    def forward(self, x: torch.Tensor) -> dict:
        # x'in boyutu: (N, C, H, W) - N=batch, C=kanallar, H=yükseklik, W=genişlik
        
        # Adım 1: Girdiyi göm
        embedded = self.input_embed(x) # -> (N, model_dim, H, W)
        
        # Adım 2: Transformer'a hazırlık
        # Transformer (batch, sequence, features) formatını bekler.
        # Haritamızı bu formata dönüştürüyoruz: (N, H*W, model_dim)
        n, c, h, w = embedded.shape
        seq = embedded.flatten(2).permute(0, 2, 1)
        
        # Adım 3: Transformer'ı çalıştır
        processed_seq = self.transformer_encoder(seq) # -> (N, H*W, model_dim)
        
        # Adım 4: Transformer çıktısını tekrar harita formatına dönüştür
        processed_map = processed_seq.permute(0, 2, 1).reshape(n, c, h, w)
        
        # Adım 5: Her akı için çıktıları hesapla ve pozitif olmalarını sağla
        # ReLU, akıların negatif olmasını engeller (fiziksel olarak anlamsız).
        fluxes = {
            'inf': torch.relu(self.head_inf(processed_map)).squeeze(1),
            'aet': torch.relu(self.head_aet(processed_map)).squeeze(1),
            'cap': torch.relu(self.head_cap(processed_map)).squeeze(1),
            'per': torch.relu(self.head_per(processed_map)).squeeze(1),
            'q1': torch.relu(self.head_q1(processed_map)).squeeze(1),
            'q2': torch.relu(self.head_q2(processed_map)).squeeze(1),
            'q3': torch.relu(self.head_q3(processed_map)).squeeze(1),
        }
        
        return fluxes