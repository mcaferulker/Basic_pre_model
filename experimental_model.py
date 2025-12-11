import torch
import torch.nn as nn
from hydrological_transformer import FluxTransformer # Yeni modülümüzü import et

class GridCellRouter(nn.Module):
    """
    Bu modül, bir DEM'den türetilmiş akış yönü haritasını kullanarak,
    bir grid üzerindeki su akışını (routing) simüle eder.
    """
    def __init__(self, flow_direction_indices, device):
        super().__init__()
        self.device = device
        
        # Akış yönü haritası (N, M) boyutunda olmalı ve her hücrenin
        # suyunun akacağı hücrenin düzleştirilmiş (flattened) indeksini içermeli.
        self.flow_direction_indices = torch.from_numpy(flow_direction_indices).long().to(device)
        self.n_cells = len(self.flow_direction_indices)

    def route_water(self, runoff_generated: torch.Tensor, iterations: int) -> torch.Tensor:
        """
        Verilen üretilmiş akış haritasını, belirtilen iterasyon sayısı kadar yönlendirir.
        
        Args:
            runoff_generated (Tensor): Yönlendirilecek suyun haritası. Boyut: (H, W)
            iterations (int): Suyun aşağı akması için yapılacak simülasyon adımı sayısı.
        
        Returns:
            Tensor: Yönlendirilmiş ve birikmiş suyun haritası (streamflow). Boyut: (H, W)
        """
        h, w = runoff_generated.shape
        
        # Gelen haritayı düz bir vektöre çevir
        current_flow = runoff_generated.flatten()
        
        # Başlangıçta biriken akış, o hücrede üretilen akış kadardır.
        accumulated_flow = current_flow.clone()

        for _ in range(iterations):
            # Her hücrenin mevcut akışını, akış yönü haritasına göre
            # bir sonraki hücreye gönder.
            # `index_add_` metodu, belirtilen indekslere değerleri ekler. Bu,
            # suyun bir sonraki hücrede "birikmesini" sağlar.
            accumulated_flow.index_add_(0, self.flow_direction_indices, current_flow)
            
            # Bir sonraki iterasyon için, akacak olan su, şu anki hücrelerin akışıdır.
            # Bu, suyun havza boyunca dalga gibi ilerlemesini sağlar.
            current_flow = accumulated_flow - current_flow 

        # Sonuçta biriken akış vektörünü tekrar (H, W) harita formatına dönüştür.
        return accumulated_flow.view(h, w)

class HybridHydroTransformer(nn.Module):
    def __init__(self, grid_shape, config, flow_direction_indices=None, device='cpu'):
        super().__init__()
        self.grid_shape = grid_shape
        self.config = config
        self.device = device

        # --- GİRDİ KANALLARI ---
        # Modele girecek tüm 2D verilerin (haritaların) sayısı.
        # 4 atmosferik girdi + 4 hazne durumu = 8 kanal.
        input_channels = 8 
        
        # --- AKI TAHMİNCİSİ ---
        # Fiziksel süreçleri taklit eden Transformer modülünü başlat.
        self.flux_predictor = FluxTransformer(input_channels=input_channels)

        # --- DEPOLAR (STOCKS) ---
        # Bunlar, her zaman adımı arasında güncellenen, suyun durumunu tutan haznelerdir.
        # `nn.Parameter` ile tanımlanırlar ancak gradyanları hesaplanmaz.
        self.snow_storage = nn.Parameter(torch.zeros(grid_shape), requires_grad=False)
        self.interception_storage = nn.Parameter(torch.zeros(grid_shape), requires_grad=False)
        self.soil_moisture_storage = nn.Parameter(torch.zeros(grid_shape), requires_grad=False)
        self.groundwater_storage = nn.Parameter(torch.zeros(grid_shape), requires_grad=False)
        
        # --- OPSİYONEL ROUTING MODÜLÜ ---
        self.routing_enabled = flow_direction_indices is not None
        if self.routing_enabled:
            print("INFO: DEM-based routing is ENABLED.")
            self.router = GridCellRouter(flow_direction_indices, self.device)
        else:
            print("INFO: DEM-based routing is DISABLED.")
            self.router = None

    def forward(self, forcing_data, src_seq_len, tgt_seq_len):
        device = forcing_data['precipitation'].device
        
        # --- DEPOLARI SIFIRLA ---
        # Her yeni tahmin dizisi için hazneleri başlangıç durumuna getir.
        self.snow_storage.data.zero_()
        self.interception_storage.data.zero_()
        self.soil_moisture_storage.data.fill_(self.config['initial_soil_moisture']) # Başlangıç nemi
        self.groundwater_storage.data.fill_(self.config['initial_groundwater']) # Başlangıç yeraltı suyu

        # --- ÇIKTI LİSTELERİ ---
        outputs = {'Q': [], 'Q_routed': [], 'q1': [], 'q2': [], 'q3': [], 'aet': [], 'physics_loss': []}

        # --- ZAMAN DÖNGÜSÜ ---
        for t in range(src_seq_len + tgt_seq_len):
            
            # --- ATMOSFERİK GİRDİLERİ AL (GÜNCELLENDİ) ---
            precip = forcing_data['precipitation'][t]
            temp = forcing_data['temperature'][t]
            pet = forcing_data['pet'][t]
            
            # DÜZELTME: LAI değişken adını config'den al.
            # Bu, 'lai_hv', 'lai_lv' veya başka bir isim kullanmamızı sağlar.
            lai_key = self.config.get('lai_variable_name', 'lai_hv') # Varsayılan olarak 'lai_hv'
            if lai_key not in forcing_data:
                raise KeyError(f"'{lai_key}' (LAI variable) not found in forcing_data. Check your var_map.")
            lai = forcing_data[lai_key][t]

            # --- BASİT FİZİKSEL SÜREÇLER ---
            # Kar/Yağmur ayrımı ve Interception (bitki örtüsü tarafından tutulma)
            rain = torch.where(temp > self.config['snow_threshold_temp'], precip, 0)
            snowfall = precip - rain
            self.snow_storage.data += snowfall

            # Yüzeydeki su (henüz akışa geçmemiş)
            water_on_surface = rain 
            # Interception'dan taşan su
            overflow_int = torch.clamp(self.interception_storage.data + water_on_surface - self.config['max_interception'], min=0)
            # Interception'a eklenen su
            to_interception = water_on_surface - overflow_int
            self.interception_storage.data += to_interception
            water_on_surface = overflow_int

            # Kar erimesi
            melt = torch.clamp(temp - self.config['snow_threshold_temp'], min=0) * self.config['snow_melt_rate']
            melt = torch.min(melt, self.snow_storage.data)
            self.snow_storage.data -= melt
            water_on_surface += melt

            # --- TRANSFORMER'A GİRDİ HAZIRLA ---
            # Modelin o anki durumunu temsil eden tüm haritaları birleştir.
            model_input = torch.stack([
                water_on_surface,
                pet,
                lai,
                temp,
                self.snow_storage.data,
                self.interception_storage.data,
                self.soil_moisture_storage.data,
                self.groundwater_storage.data
            ], dim=0).unsqueeze(0) # (C, H, W) -> (1, C, H, W) batch boyutu ekle

            # --- AKILARI TAHMİN ET ---
            fluxes = self.flux_predictor(model_input)
            
            # --- SU BÜTÇESİNİ GÜNCELLE ---
            # Transformer'ın tahminlerini kullanarak hazneler arasında suyu hareket ettir.
            # Her akının, mevcut su miktarını aşmamasını garanti altına al (fiziksel tutarlılık).
            
            # 1. Yüzeyden ve Interception'dan Buharlaşma + Sızma
            available_for_aet_inf = self.interception_storage.data + water_on_surface
            fluxes['aet'] = torch.min(fluxes['aet'], available_for_aet_inf * pet) # PET ile ölçekle
            fluxes['inf'] = torch.min(fluxes['inf'], available_for_aet_inf - fluxes['aet'])
            
            # Buharlaşmayı ve sızmayı yüzey ve interception'dan orantılı olarak çıkar
            frac_aet_int = self.interception_storage.data / available_for_aet_inf
            frac_aet_surf = water_on_surface / available_for_aet_inf
            self.interception_storage.data -= fluxes['aet'] * frac_aet_int
            water_on_surface -= fluxes['aet'] * frac_aet_surf
            
            frac_inf_int = self.interception_storage.data / (available_for_aet_inf - fluxes['aet'])
            frac_inf_surf = water_on_surface / (available_for_aet_inf - fluxes['aet'])
            self.interception_storage.data -= fluxes['inf'] * frac_inf_int
            water_on_surface -= fluxes['inf'] * frac_inf_surf
            
            # 2. Yüzey Altı Haznesini Güncelle
            fluxes['per'] = torch.min(fluxes['per'], self.soil_moisture_storage.data) # Perkole olabilecek su
            fluxes['cap'] = torch.min(fluxes['cap'], self.soil_moisture_storage.data) # Kılcal yükselim olabilecek su
            self.soil_moisture_storage.data += fluxes['inf'] - fluxes['per'] - fluxes['cap']

            # 3. Yeraltı Suyu Haznesini Güncelle
            self.groundwater_storage.data += fluxes['per']

            # 4. Akışları Hesapla
            fluxes['q1'] = torch.min(fluxes['q1'], water_on_surface) # Yüzey akışı, yüzeydeki sudan fazla olamaz
            fluxes['q2'] = torch.min(fluxes['q2'], self.soil_moisture_storage.data) # Yüzey altı akışı
            fluxes['q3'] = torch.min(fluxes['q3'], self.groundwater_storage.data) # Yeraltı suyu akışı

            # Akışları ait oldukları haznelerden çıkar
            self.soil_moisture_storage.data -= fluxes['q2']
            self.groundwater_storage.data -= fluxes['q3']

            # Toplam Akış (Q)
            total_runoff = fluxes['q1'] + fluxes['q2'] + fluxes['q3']
            
            # --- OPSİYONEL ROUTING ADIMI ---
            if self.routing_enabled:
                total_runoff_routed = self.router.route_water(
                    runoff_generated=total_runoff,
                    iterations=self.config.get('routing_iterations', 10) # Konfigürasyondan al veya 10 varsay
                )
            else:
                # Routing devre dışıysa, yönlendirilmiş akış üretilen akışla aynıdır.
                total_runoff_routed = total_runoff
            
            # Tüm haznelerin negatif olmamasını sağla
            self.snow_storage.data.clamp_(min=0)
            self.interception_storage.data.clamp_(min=0)
            self.soil_moisture_storage.data.clamp_(min=0)
            self.groundwater_storage.data.clamp_(min=0)

            # --- ÇIKTILARI KAYDET ---
            if t >= src_seq_len:
                outputs['Q'].append(total_runoff)
                outputs['Q_routed'].append(total_runoff_routed) # Yönlendirilmiş akış
                outputs['q1'].append(fluxes['q1']); outputs['q2'].append(fluxes['q2']); outputs['q3'].append(fluxes['q3'])
                outputs['aet'].append(fluxes['aet'])
                # outputs['physics_loss'].append(...) # Daha gelişmiş bir su bütçesi kaybı eklenebilir
        
        # Listeleri tensörlere dönüştür
        for key in outputs:
            if len(outputs[key]) > 0: outputs[key] = torch.stack(outputs[key], dim=0)
        
        # 'total_runoff' anahtarını diğer kodların beklemesi üzerine ekle
        if 'Q' in outputs: outputs['total_runoff'] = outputs['Q_routed'] if self.routing_enabled else outputs['Q']
        
        return outputs