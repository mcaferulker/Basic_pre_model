import torch
import torch.nn as nn
import torch.nn.functional as F
from hydrological_transformer import FluxTransformer

class HybridHydroTransformer(nn.Module):
    """
    "Hibrit Kısıtlama" modelinin, analiz için tüm içsel durumları ve
    akıları da döndüren gelişmiş versiyonu.
    """
    def __init__(self, grid_shape, config, device='cpu'):
        super().__init__()
        self.grid_shape = grid_shape
        self.config = config
        self.device = device
        self.use_lai = config.get('use_lai', True)
        input_channels = 8 if self.use_lai else 7
        self.flux_predictor = FluxTransformer(input_channels=input_channels)

    def forward(self, forcing_data: dict, seq_len: int, warmup_steps: int):
        # --- Durum Depolarını Başlat ---
        snow = torch.zeros(self.grid_shape, device=self.device)
        interception = torch.zeros(self.grid_shape, device=self.device)
        soil_moisture = torch.full(self.grid_shape, self.config['initial_soil_moisture'], device=self.device)
        groundwater = torch.full(self.grid_shape, self.config['initial_groundwater'], device=self.device)
        
        # GÜNCELLENDİ: Kaydedilecek tüm değişkenler için listeler oluştur
        outputs = {
            'Q_total': [], 'physics_loss': [], 'snow': [], 'soil_moisture': [],
            'groundwater': [], 'interception': [], 'AET': [], 'q1': [], 'q2': [], 'q3': []
        }

        for t in range(seq_len):
            # ... (Girdi hazırlama ve akı talebi kısmı aynı kalır) ...
            precip, temp, pet = forcing_data['precipitation'][t], forcing_data['temperature'][t], forcing_data['pet'][t]
            rain = torch.where(temp > self.config['snow_threshold_temp'], precip, 0.0)
            snowfall = precip - rain
            snow += snowfall
            melt = torch.min(snow, torch.clamp(temp - self.config['snow_threshold_temp'], min=0.0) * self.config['snow_melt_rate'])
            snow -= melt
            water_on_surface = rain + melt
            model_input_tensors = [water_on_surface, pet, temp, snow, interception, soil_moisture, groundwater]
            if self.use_lai: model_input_tensors.append(forcing_data['lai_hv'][t])
            model_input = torch.stack([t.squeeze() for t in model_input_tensors], dim=0).unsqueeze(0)
            flux_demands = self.flux_predictor(model_input)
            
            q1_d, q2_d, q3_d = flux_demands['q1'].squeeze(), flux_demands['q2'].squeeze(), flux_demands['q3'].squeeze()
            inf_d, per_d, aet_d, cap_d = flux_demands['inf'].squeeze(), flux_demands['per'].squeeze(), flux_demands['aet'].squeeze(), flux_demands['cap'].squeeze()
            
            # --- HİBRİT SU BÜTÇESİ ---
            aet_demand_capped = torch.min(aet_d, pet)
            real_aet = torch.min(aet_demand_capped, interception + soil_moisture) # AET'nin gerçek değeri

            total_surface_demand = q1_d + inf_d
            surface_scaling_factor = torch.min(torch.ones_like(water_on_surface), water_on_surface / (total_surface_demand + 1e-9))
            q1 = q1_d * surface_scaling_factor
            inf = inf_d * surface_scaling_factor
            
            temp_interception = interception + water_on_surface - inf - q1 - aet_demand_capped
            temp_soil = soil_moisture + inf - per_d - q2_d + cap_d
            temp_groundwater = groundwater + per_d - q3_d - cap_d

            physics_loss = torch.mean(F.relu(-temp_interception)**2) + torch.mean(F.relu(-temp_soil)**2) + torch.mean(F.relu(-temp_groundwater)**2)

            interception = F.relu(temp_interception)
            soil_moisture = F.relu(temp_soil)
            groundwater = F.relu(temp_groundwater)

            # total_runoff'u, gradyan akışını korumak için orijinal taleplerden hesapla
            total_runoff = q1_d + q2_d + q3_d

            if t >= warmup_steps:
                # GÜNCELLENDİ: Tüm içsel durumları ve akıları kaydet
                outputs['Q_total'].append(total_runoff)
                outputs['physics_loss'].append(physics_loss)
                outputs['snow'].append(snow)
                outputs['soil_moisture'].append(soil_moisture)
                outputs['groundwater'].append(groundwater)
                outputs['interception'].append(interception)
                outputs['AET'].append(real_aet)
                outputs['q1'].append(q1)
                outputs['q2'].append(q2_d) # q2 ve q3 için de orijinal talepleri kaydedelim
                outputs['q3'].append(q3_d)
        
        # --- Çıktıları Hazırla ---
        final_outputs = {}
        for key, tensor_list in outputs.items():
            if not tensor_list: continue
            stacked_tensor = torch.stack(tensor_list, dim=0)
            final_outputs[key] = stacked_tensor.squeeze()
        
        return final_outputs