import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import xarray as xr
import torch.nn.functional as F

from data_loader import load_and_process_data_from_folders
from experimental_model import HybridHydroTransformer
from losses import KGELoss # YENİ: KGE Kaybını import et

# =============================================================================
# 1. DEĞERLENDİRME VE GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================
def calculate_metrics_robust(pred_np: np.ndarray, obs_np: np.ndarray) -> dict:
    metrics = {'nse': np.nan, 'kge': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
    epsilon = 1e-9
    valid_indices = np.isfinite(pred_np) & np.isfinite(obs_np)
    pred_flat = pred_np[valid_indices].flatten()
    obs_flat = obs_np[valid_indices].flatten()
    if len(obs_flat) < 2 or np.std(obs_flat) < epsilon: return metrics
    try:
        nse_denom = np.sum((obs_flat - np.mean(obs_flat)) ** 2)
        if nse_denom > epsilon: metrics['nse'] = 1 - (np.sum((pred_flat - obs_flat) ** 2) / nse_denom)
        mean_obs, mean_pred = np.mean(obs_flat), np.mean(pred_flat)
        std_obs, std_pred = np.std(obs_flat), np.std(pred_flat)
        if std_obs > epsilon and abs(mean_obs) > epsilon:
            metrics['r'] = np.corrcoef(pred_flat, obs_flat)[0, 1]
            metrics['alpha'] = std_pred / std_obs
            metrics['beta'] = mean_pred / mean_obs
            if np.isnan(metrics['r']): metrics['r'] = 0
            metrics['kge'] = 1 - np.sqrt((metrics['r'] - 1)**2 + (metrics['alpha'] - 1)**2 + (metrics['beta'] - 1)**2)
    except Exception: pass
    return metrics

def save_results_and_visualize(pred_tensor: torch.Tensor, obs_tensor: torch.Tensor, time_axis, var_name: str):
    # Şekle göre karar verelim
    def to_series(x: torch.Tensor) -> np.ndarray:
        arr = x.cpu().numpy()
        if arr.ndim == 3:   # (time, lat, lon)
            return arr.mean(axis=(1, 2))
        elif arr.ndim == 2: # (time, 1) veya (time, nfeat) gibi
            return arr.mean(axis=1)
        else:               # (time,)
            return arr

    pred_series_mm = to_series(pred_tensor)
    obs_series_mm  = to_series(obs_tensor)

    # Bağımsız kontrol için uzunluk eşitliği
    assert len(pred_series_mm) == len(obs_series_mm), f"len pred={len(pred_series_mm)}, obs={len(obs_series_mm)}"

    series_metrics = calculate_metrics_robust(pred_series_mm, obs_series_mm)

    plt.style.use('ggplot')
    plt.figure(figsize=(15, 6))
    plt.plot(time_axis, obs_series_mm, label='Observed', color='black', lw=2)
    plt.plot(time_axis, pred_series_mm, label='Predicted', color='dodgerblue', ls='--')
    plt.title(f'Basin Average {var_name} (NSE: {series_metrics['nse']:.3f} | KGE: {series_metrics['kge']:.3f})')
    plt.xlabel('Date'); plt.ylabel('Daily Average Value (mm/day)')
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# YENİ FONKSİYON: NetCDF kaydetme
def save_outputs_to_netcdf(outputs: dict, time_coords, filename="model_outputs.nc"):
    """
    Modelin tüm çıktılarını (zaman, lat, lon) boyutlarıyla bir NetCDF dosyasına kaydeder.
    """
    data_vars = {}
    for var_name, tensor in outputs.items():
        if tensor.ndim == 3:  # Sadece (time, lat, lon) şeklindeki tensörleri al
            dims = ('time', 'lat', 'lon')
            coords = {'time': time_coords, 'lat': np.arange(tensor.shape[1]), 'lon': np.arange(tensor.shape[2])}
            data_vars[var_name] = xr.DataArray(tensor.numpy(), dims=dims, coords=coords)

    if not data_vars:
        print("Warning: No 3D data variables found to save to NetCDF.")
        return

    ds = xr.Dataset(data_vars)
    try:
        ds.to_netcdf(filename)
        print(f"Model outputs successfully saved to '{filename}'")
    except Exception as e:
        print(f"Error saving to NetCDF: {e}")

class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        ey_t =  y_true - y_pred
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

# =============================================================================
# 2. ANA EĞİTİM FONKSİYONU
# =============================================================================
def train_experiment():
    ROOT_DATA_DIR = "data"
    FILE_NAME_MAP = {
        'precipitation': 'pre', 'temperature': 'temp', 'pet': 'pev_runoff',
        'runoff_observed_map': 'pev_runoff'
    }
    VAR_MAP = {
        'precipitation': 'tp', 'temperature': 't2m', 'pet': 'pev',
        'runoff_observed_map': 'ro'
    }
    TIME_SPLITS = {
        'train_start': '2020-02-01', 'train_end': '2023-12-31',
        'test_start': '2024-01-01', 'test_end': '2024-12-31', 'warmup_days': 30
    }
    HYBRID_CONFIG = {
        "max_interception": 3.0, "snow_threshold_temp": 0.5,
        "snow_melt_rate": 1.0, "initial_soil_moisture": 100.0,
        "initial_groundwater": 50.0, "use_lai": False
    }
    NUM_EPOCHS, LEARNING_RATE = 50, 0.001
    LAMBDA_PHYSICS = 1500.0 # Daha agresif ağırlık

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_data = load_and_process_data_from_folders(
        ROOT_DATA_DIR,
        FILE_NAME_MAP,
        VAR_MAP,
        TIME_SPLITS,
        convert_to_mm=True
    )

    grid_shape = split_data['attributes']['grid_shape']
    model = HybridHydroTransformer(grid_shape, HYBRID_CONFIG, device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    # GÜNCELLENDİ: Artık MSELoss yerine KGELoss kullanıyoruz.
    kge_loss_fn = KGELoss().to(device)
    mse_loss_fn = nn.MSELoss() #LogCoshLoss()
    KGE_WEIGHT = 0.7

    train_data = {k: v.to(device) for k, v in split_data['train'].items() if isinstance(v, torch.Tensor)}
    test_data = {k: v.to(device) for k, v in split_data['test'].items() if isinstance(v, torch.Tensor)}

    # --- EN İYİ MODELİ İZLEMEK İÇİN DEĞİŞKENLER ---
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    best_train_outputs_for_plot = None

    print("\n--- Starting Training with KGE Loss and Physics-Informed Model ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        # --- EĞİTİM ADIMI ---
        train_outputs = model(train_data, seq_len=len(train_data['precipitation']), warmup_steps=split_data['train']['warmup_steps'])
        pred_train_maps = train_outputs['Q_total']
        obs_train_maps = train_data['runoff_observed_map'][split_data['train']['warmup_steps']:]
        # Ana kaybı iki bileşenin ağırlıklı toplamı olarak hesapla
        kge_loss = kge_loss_fn(pred_train_maps, obs_train_maps)
        mse_loss = mse_loss_fn(pred_train_maps, obs_train_maps)

        main_loss = KGE_WEIGHT * kge_loss + (1 - KGE_WEIGHT) * mse_loss

        physics_loss = torch.mean(train_outputs.get('physics_loss', torch.tensor(0.0, device=device)))
        total_train_loss = main_loss + LAMBDA_PHYSICS * physics_loss
        # Eğitim döngüsünde (epoch içinde), backward'dan önce:
        if epoch == 0:  # bir kere yazdırmak kâfi
          print("train pred shape:", pred_train_maps.shape, "obs shape:", obs_train_maps.shape)
        total_train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- DOĞRULAMA (VALIDATION) ADIMI (Her epoch sonunda test seti üzerinde) ---
        model.eval()
        with torch.no_grad():
            val_outputs = model(test_data, seq_len=len(test_data['precipitation']), warmup_steps=split_data['test']['warmup_steps'])
            pred_val_maps = val_outputs['Q_total']
            obs_val_maps = test_data['runoff_observed_map'][split_data['test']['warmup_steps']:]

            val_kge_loss = kge_loss_fn(pred_val_maps, obs_val_maps)
            val_mse_loss = mse_loss_fn(pred_val_maps, obs_val_maps)
            val_main_loss = KGE_WEIGHT * val_kge_loss + (1 - KGE_WEIGHT) * val_mse_loss

            val_physics_loss = torch.mean(val_outputs.get('physics_loss', torch.tensor(0.0, device=device)))
            total_val_loss = val_main_loss + LAMBDA_PHYSICS * val_physics_loss
        assert pred_train_maps.shape[0] == obs_train_maps.shape[0], "train time len mismatch"
        assert pred_val_maps.shape[0]   == obs_val_maps.shape[0],   "val time len mismatch"
        # --- EN İYİ MODELİ KAYDETME ---
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            # En iyi anın eğitim çıktısını da çizdirmek için kaydet
            best_train_outputs_for_plot = {'Q_total': pred_train_maps.cpu().detach()}
            print(f"  ---> New best model found at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

        scheduler.step(total_val_loss)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {total_train_loss.item():.4f} (KGE: {kge_loss.item():.3f}, MSE: {mse_loss.item():.3f}), Val Loss: {total_val_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.1e}")

    print("\n--- Training Finished ---")
    # --- EN İYİ MODELİN EĞİTİM PERFORMANSINI GÖRSELLEŞTİR ---
    if best_train_outputs_for_plot:
        print(f"\n--- Plotting Training Performance of Best Model (from Epoch {best_epoch}) ---")
        obs_train_final = train_data['runoff_observed_map'][split_data['train']['warmup_steps']:].cpu()
        save_results_and_visualize(best_train_outputs_for_plot['Q_total'], obs_train_final, split_data['train']['dates'], "BEST MODEL - TRAIN SET")

    # --- EN İYİ MODEL İLE FİNAL DEĞERLENDİRME ---
    print("\n--- Final Evaluation on Test Set using Best Model ---")
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model was saved. Evaluating with the final model.")

    model.eval()
    with torch.no_grad():
        final_outputs = model(test_data, seq_len=len(test_data['precipitation']), warmup_steps=split_data['test']['warmup_steps'])

    final_pred_mm = final_outputs['Q_total'].cpu()
    final_obs_mm = test_data['runoff_observed_map'][split_data['test']['warmup_steps']:].cpu()
    assert final_pred_mm.shape[0]   == final_obs_mm.shape[0],   "test time len mismatch"
    assert len(split_data['test']['dates']) == final_pred_mm.shape[0], "date axis mismatch"
    # Final Test Grafiğini Çizdir
    save_results_and_visualize(final_pred_mm, final_obs_mm, split_data['test']['dates'], "BEST MODEL - TEST SET")

    # YENİ: Final Çıktılarını NetCDF Olarak Kaydet
    save_outputs_to_netcdf(
        {k: v.cpu() for k, v in final_outputs.items()},
        time_coords=split_data['test']['dates'],
        filename="best_model_test_outputs.nc"
    )
    with torch.no_grad():
      ps = final_pred_mm if 'final_pred_mm' in locals() else pred_val_maps.cpu()
      os = final_obs_mm if 'final_obs_mm' in locals() else obs_val_maps.cpu()

      def to_series_np(x):
          arr = x.numpy()
          if arr.ndim == 3:
              return arr.mean(axis=(1,2))
          elif arr.ndim == 2:
              return arr.mean(axis=1)
          else:
              return arr

      ps_np = to_series_np(ps)
      os_np = to_series_np(os)

      from math import isfinite
      def nse_np(p, o):
          mask = np.isfinite(p) & np.isfinite(o)
          p, o = p[mask], o[mask]
          return 1 - np.sum((p - o)**2) / np.sum((o - o.mean())**2)
      def kge_np(p, o, eps=1e-6):
          mask = np.isfinite(p) & np.isfinite(o)
          p, o = p[mask], o[mask]
          vx, vy = p - p.mean(), o - o.mean()
          r = np.sum(vx*vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)) + eps)
          alpha = p.std() / (o.std()+eps)
          beta = p.mean() / (o.mean()+eps)
          return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

      print("Sanity NSE:", nse_np(ps_np, os_np))
      print("Sanity KGE:", kge_np(ps_np, os_np))

if __name__ == "__main__":
    train_experiment()