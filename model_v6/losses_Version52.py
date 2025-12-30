import torch
import torch.nn as nn

class KGELoss(nn.Module):
    """
    Kling-Gupta Efficiency (KGE) metriğini bir kayıp fonksiyonu olarak uygular.
    Modeli, korelasyonu (r), varyans oranını (alpha) ve ortalama oranını (beta)
    aynı anda optimize etmeye zorlar. Bu, pikleri ve taban akışını daha iyi
    yakalaması için güçlü bir sinyaldir.
    """
    def __init__(self, epsilon: float = 1e-6):
        super(KGELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ortalama ve standart sapmaları hesapla
        mean_pred = torch.mean(y_pred)
        mean_true = torch.mean(y_true)
        std_pred = torch.std(y_pred)
        std_true = torch.std(y_true)

        # Korelasyon (r)
        vx = y_pred - mean_pred
        vy = y_true - mean_true
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.epsilon)

        # Varyans Oranı (alpha)
        alpha = std_pred / (std_true + self.epsilon)

        # Ortalama Oranı (beta)
        beta = mean_pred / (mean_true + self.epsilon)

        # KGE'yi hesapla
        kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # KGE'yi maksimize etmek, (1 - KGE)'yi minimize etmek demektir.
        return 1 - kge