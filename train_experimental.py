import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
import hydroeval as he

# Proje dosyalarımızdan gerekli fonksiyonları ve sınıfları import edelim
from data_loader import load_and_process_netcdf_data
from experimental_model import ExperimentalHydrologyModel

# =============================================================================
# 1. KAYIP FONKSİYONU SINIFLARI
#    Modeli eğitmek için kullanılacak özel kayıp fonksiyonları burada tanımlanır.
# =============================================================================

class NSeriesLoss(nn.Module):
    """Zaman serileri arasındaki (1 - NSE) değerini kayıp olarak hesaplar."""
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        # Haritaların (time, lat, lon) uzamsal ortalamasını alarak zaman serisi oluştur
        pred_series = torch.mean(predictions, dim=[1, 2])
        obs_series = torch.mean(observations, dim=[1, 2])

        mean_obs = torch.mean(obs_series)
        numerator = torch.sum((obs_series - pred_series) ** 2)
        denominator = torch.sum((obs_series - mean_obs) ** 2) + self.epsilon
        nse = 1.0 - (numerator / denominator)
        return 1.0 - nse

class SPAEFLoss(nn.Module):
    """(1 - SPAEF) metriğini kayıp olarak hesaplar. Mekansal desen doğruluğunu hedefler."""
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred_maps: torch.Tensor, obs_maps: torch.Tensor) -> torch.Tensor:
        # Girdiler (time, lat, lon) -> Düzleştir (time, n_locations)
        pred_flat = pred_maps.flatten(start_dim=1)
        obs_flat = obs_maps.flatten(start_dim=1)

        # PyTorch ile Pearson Korelasyonu (r)
        vx = pred_flat - torch.mean(pred_flat, dim=0)
        vy = obs_flat - torch.mean(obs_flat, dim=0)
        r = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + self.epsilon)

        # Değişkenlik (alpha) ve Bias (beta)
        alpha = torch.std(pred_flat) / (torch.std(obs_flat) + self.epsilon)
        beta = torch.mean(pred_flat) / (torch.mean(obs_flat) + self.epsilon)

        spaef = 1.0 - torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        return 1.0 - spaef

# =============================================================================
# 2. DEĞERLENDİRME METRİKLERİ HESAPLAYICI
# =============================================================================
def calculate_metrics_robust(pred_np: np.ndarray, obs_np: np.ndarray) -> dict:
    """Tüm metrikleri ve KGE bileşenlerini numpy kullanarak güvenli bir şekilde hesaplar."""
    metrics = {'nse': np.nan, 'kge': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
    epsilon = 1e-6
    
    pred_flat, obs_flat = pred_np.flatten(), obs_np.flatten()

    # Gözlem verisinin varyansını kontrol et
    if np.std(obs_flat) < epsilon:
        return metrics

    try:
        # NSE
        metrics['nse'] = 1 - (np.sum((pred_flat - obs_flat) ** 2) / np.sum((obs_flat - np.mean(obs_flat)) ** 2))
        
        # KGE Bileşenleri
        metrics['r'] = np.corrcoef(pred_flat, obs_flat)[0, 1]
        metrics['alpha'] = np.std(pred_flat) / np.std(obs_flat)
        metrics['beta'] = np.mean(pred_flat) / np.mean(obs_flat)
        
        # KGE
        metrics['kge'] = 1 - np.sqrt((metrics['r'] - 1)**2 + (metrics['alpha'] - 1)**2 + (metrics['beta'] - 1)**2)
    except Exception:
        pass # Hata durumunda NaN olarak kalır
        
    return metrics

# =============================================================================
# 3. ANA EĞİTİM VE DEĞERLENDİRME FONKSİYONU
# =============================================================================
def train_experiment():
    # ==================================================================
    # --- KONFİGÜRASYON BÖLÜMÜ (BURAYI DÜZENLEYİN) ---
    # ==================================================================
    file_paths = {
        'precipitation': 'pre_2023.nc', 'temperature': 'temp_2023.nc',
        'pet': 'pev_runoff_2023.nc', 'lai_hv': 'low_high_LAI_2023.nc',
        'lai_lv': 'low_high_LAI_2023.nc', 'runoff_observed_map': 'pev_runoff_2023.nc'
    }
    var_map = {
        'precipitation': 'tp', 'temperature': 't2m', 'pet': 'pev',
        'lai_hv': 'lai_hv', 'lai_lv': 'lai_lv', 'runoff_observed_map': 'ro'
    }
    target_grid_key = 'runoff_observed_map'
    time_range = None

    SRC_SEQ_LEN, TGT_SEQ_LEN, NUM_EPOCHS, LEARNING_RATE = 30, 180, 5, 0.001

    # --- ESNEK KAYIP AĞIRLIKLARI ---
    # Modelin neye odaklanacağını bu ağırlıkları değiştirerek yönetebilirsiniz.
    evaluation_config = {
        'total_runoff': {
            'observed_key': 'runoff_observed_map',
            'lambda_mse': 0.5,        # Piksel bazında doğruluk (değer)
            'lambda_spaef': 1.0,      # Mekansal desen doğruluğu (harita dokusu)
            'lambda_nse_series': 0.5  # Zamansal desen doğruluğu (günlük ortalama)
        },
        # GELECEK İÇİN NOT: Başka bir değişkeni (örn: AET) de eğitime dahil etmek isterseniz,
        # elinizde 'aet_observed.nc' gibi bir gözlem verisi olduğunda buraya ekleyebilirsiniz:
        #
        # 'aet': {
        #     'observed_key': 'aet_observed_map', # Bu anahtarı file_paths'e de ekleyin
        #     'lambda_mse': 0.3,
        #     'lambda_spaef': 0.5,
        #     'lambda_nse_series': 0.2
        # },
    }
    LAMBDA_PHYSICS = 0.1

    # --- Fiziksel Model Konfigürasyonu ---
    physics_config = {
        "snow_threshold_temp": 0.5, "snow_melt_rate": 0.005,
        "max_infiltration_rate": 0.01, 'soil_field_capacity': 0.25, 'soil_depth': 0.5
    }
    # ==================================================================

    # --- Veri Yükleme ve Model Başlatma ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    all_data = load_and_process_netcdf_data(file_paths, var_map, target_grid_key, time_range)

    grid_shape = all_data['precipitation'].shape[1:]
    model = ExperimentalHydrologyModel(grid_shape=grid_shape, physics_config=physics_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Kayıp fonksiyonlarını başlat
    mse_loss_fn = nn.MSELoss()
    nse_series_loss_fn = NSeriesLoss().to(device)
    spaef_loss_fn = SPAEFLoss().to(device)

    forcing_data = {k: v.to(device) for k, v in all_data.items() if isinstance(v, torch.Tensor)}

    # --- Eğitim Döngüsü ---
    print("\n--- Starting Training Loop ---")
    loss_history = []
    for epoch in range(NUM_EPOCHS):
        model.train(); optimizer.zero_grad()
        outputs = model(forcing_data, SRC_SEQ_LEN, TGT_SEQ_LEN)

        total_loss = torch.tensor(0.0, device=device)

        # evaluation_config'e göre dinamik olarak kayıpları hesapla
        for var_name, config in evaluation_config.items():
            if var_name not in outputs or config['observed_key'] not in all_data: continue

            pred_maps = outputs[var_name]
            obs_maps = all_data[config['observed_key']][SRC_SEQ_LEN : SRC_SEQ_LEN + len(pred_maps)].to(device)

            total_loss += config.get('lambda_mse', 0.0) * mse_loss_fn(pred_maps, obs_maps)
            total_loss += config.get('lambda_spaef', 0.0) * spaef_loss_fn(pred_maps, obs_maps)
            total_loss += config.get('lambda_nse_series', 0.0) * nse_series_loss_fn(pred_maps, obs_maps)

        total_loss += LAMBDA_PHYSICS * outputs.get("physics_loss_avg", 0.0)

        total_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        loss_history.append(total_loss.item())
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Total Loss: {total_loss.item():.6f}")

    print("\n--- Training Finished ---\n")

    # --- Final Değerlendirme ve Görselleştirme ---
    print("--- Final Evaluation & Visualization ---"); 
    model.eval()
    with torch.no_grad():
        final_len = len(all_data['daily_dates']) - SRC_SEQ_LEN
        final_outputs = model(forcing_data, SRC_SEQ_LEN, final_len)

        # --- DÜZELTME: Metrikleri burada merkezi olarak hesapla ---
        # Grafiklerde ve loglarda kullanmak üzere metrikleri bir sözlükte sakla
        final_metrics = {}
        for var_name, config in evaluation_config.items():
            if var_name not in final_outputs: continue
            
            print(f"\n--- Metrics for: {var_name.upper()} ---")
            pred_maps = final_outputs[var_name].cpu().numpy()
            obs_maps = all_data[config['observed_key']][SRC_SEQ_LEN:].cpu().numpy()
            
            # Mekansal metrikler
            map_metrics = calculate_metrics_robust(pred_maps, obs_maps)
            print(f"  Spatial (Map-based): NSE: {map_metrics['nse']:.3f} | KGE: {map_metrics['kge']:.3f}")
            
            # Zamansal metrikler
            pred_series = np.mean(pred_maps, axis=(1, 2))
            obs_series = np.mean(obs_maps, axis=(1, 2))
            series_metrics = calculate_metrics_robust(pred_series, obs_series)
            print(f"  Temporal (Time-Series): NSE: {series_metrics['nse']:.3f} | KGE: {series_metrics['kge']:.3f} (r={series_metrics['r']:.2f}, α={series_metrics['alpha']:.2f}, β={series_metrics['beta']:.2f})")
            
            final_metrics[var_name] = {'map': map_metrics, 'series': series_metrics}
        
        # =====================================================================
        # 5. FİNAL ÇIKTILARINI NETCDF OLARAK KAYDETME (DÜZELTİLDİ)
        # =====================================================================
        print("\n--- Saving Final Outputs to NetCDF ---")
        try:
            with xr.open_dataset(file_paths[target_grid_key]) as ref_ds:
                ref_var_name = var_map.get(target_grid_key)
                if ref_var_name not in ref_ds:
                    raise KeyError(f"Reference variable '{ref_var_name}' not found in '{file_paths[target_grid_key]}'")
                
                # --- AKILLI BOYUT BULMA ---
                # Boyut adlarının ne olduğunu varsaymak yerine, içinde 'lat' veya 'lon' geçenleri bul.
                # Bu, 'latitude', 'longitude', 'lat', 'lon' gibi farklı isimlendirmelere karşı çalışır.
                lat_dim_name = next((dim for dim in ref_ds[ref_var_name].dims if 'lat' in dim.lower()), None)
                lon_dim_name = next((dim for dim in ref_ds[ref_var_name].dims if 'lon' in dim.lower()), None)

                if not lat_dim_name or not lon_dim_name:
                    raise ValueError(f"Could not automatically find latitude/longitude dimension names in reference variable.")

                print(f"  -> Found dimension names: lat='{lat_dim_name}', lon='{lon_dim_name}'")

                lat_coords = ref_ds[lat_dim_name]
                lon_coords = ref_ds[lon_dim_name]
            
            time_coords = all_data['daily_dates'][SRC_SEQ_LEN:]

            for name, tensor in final_outputs.items():
                if not isinstance(tensor, torch.Tensor) or len(tensor.shape) < 3:
                    continue  # Sadece harita tensörlerini (time, lat, lon) kaydet

                # Düzeltme: DataArray'i, dinamik olarak bulduğumuz doğru boyut adlarıyla oluştur.
                output_da = xr.DataArray(
                    data=tensor.cpu().numpy(),
                    coords={
                        "time": time_coords, 
                        lat_dim_name: lat_coords,
                        lon_dim_name: lon_coords
                    },
                    dims=["time", lat_dim_name, lon_dim_name],
                    name=name
                )
                filename = f"predicted_{name}.nc"
                output_da.to_netcdf(filename)
                print(f"  -> Saved: {filename}")

        except Exception as e:
            print(f"  -> ERROR during NetCDF saving: {e}")
        
        # --- Grafik Çizimleri ---
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 5)); plt.plot(range(1, NUM_EPOCHS + 1), loss_history, marker='o')
        plt.title('Total Loss vs. Epochs'); plt.xlabel('Epoch'); plt.ylabel('Total Loss')
        plt.xticks(range(0, NUM_EPOCHS + 1, 5)); plt.tight_layout(); plt.show()

        var_to_plot = next(iter(evaluation_config))
        pred_series_plot = np.mean(final_outputs[var_to_plot].cpu().numpy(), axis=(1, 2))
        obs_series_plot = np.mean(all_data[evaluation_config[var_to_plot]['observed_key']][SRC_SEQ_LEN:].cpu().numpy(), axis=(1, 2))
        time_axis = all_data['daily_dates'][SRC_SEQ_LEN:]
        
        # DÜZELTME: Önceden hesaplanmış, skaler metrik değerlerini kullan
        nse_val = final_metrics[var_to_plot]['series']['nse']
        kge_val = final_metrics[var_to_plot]['series']['kge']

        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, obs_series_plot, label='Observed', color='black', lw=2)
        plt.plot(time_axis, pred_series_plot, label='Predicted', color='dodgerblue', ls='--')
        plt.title(f'Basin Average {var_to_plot.replace("_", " ").title()} (NSE: {nse_val:.3f} | KGE: {kge_val:.3f})')
        plt.xlabel('Date'); plt.ylabel('Daily Average Value (m/day)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    train_experiment()