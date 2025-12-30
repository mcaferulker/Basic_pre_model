import torch
import torch.nn as nn
import torch.nn.functional as F # YENİ: Eksik olan import eklendi

class FluxTransformer(nn.Module):
    """
    Girdi forcing verilerini alır ve 7 ana hidrolojik akı için tahmin haritaları üretir.
    """
    def __init__(self, input_channels, model_dim=128, n_head=4, n_layers=3):
        super().__init__()
        
        self.input_embed = nn.Conv2d(input_channels, model_dim, kernel_size=1)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_head, dim_feedforward=model_dim * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.flux_names = ['q1', 'q2', 'q3', 'inf', 'per', 'aet', 'cap']
        self.output_heads = nn.ModuleDict({
            name: nn.Conv2d(model_dim, 1, kernel_size=1) for name in self.flux_names
        })

    def forward(self, x: torch.Tensor) -> dict:
        """
        Girdi tensörünü işler ve her akı için bir tahmin içeren bir sözlük döndürür.
        """
        embedded = self.input_embed(x)
        n, c, h, w = embedded.shape
        seq = embedded.flatten(2).permute(0, 2, 1)
        
        processed_seq = self.transformer_encoder(seq)
        processed_map = self.transformer_encoder(seq).permute(0, 2, 1).reshape(n, c, h, w)
        
        # GÜNCELLENDİ: Artık bir oran değil, 'mm/gün' cinsinden mutlak bir talep üretiyoruz.
        # Softplus, talebin her zaman pozitif olmasını sağlar.
        flux_demands = {
            name: F.softplus(head(processed_map)) 
            for name, head in self.output_heads.items()
        }
        
        return flux_demands