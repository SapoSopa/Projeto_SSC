"""
feature_extraction.py
Módulo de extração de características de sinais fisiológicos
Inclui carregamento de dados processados, extração de características
em diferentes domínios e salvamento dos resultados
"""

import os
import json
import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple
import matplotlib.pyplot as plt 


# ---------------------- #
# 1- Carregar sinal salvo
# ---------------------- #

def load_signal_processado(filepath_npz: str) -> Tuple[np.ndarray, dict]:

    """
    Carrega um sinal previamente salvo no formato .npz e retorna metadados
    """

    if not os.path.exists(filepath_npz):
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath_npz}")

    data = np.load(filepath_npz, allow_pickle=True)
    sinal = data['sinal']
    metadata = {
        'fs': int(data['fs']) if 'fs' in data else 100,
        'shape': tuple(data['shape']) if 'shape' in data else sinal.shape,
        'ecg_id': int(data['ecg_id']) if 'ecg_id' in data else -1,
        'timestamp': str(data['timestamp']) if 'timestamp' in data else "unknown"
    }
    return sinal, metadata


# ------------------------------- #
# 2- Extração de características
# ------------------------------- #

def extract_time_features(signal: np.ndarray) -> Dict[str, float]:

    """
    Extrai características do domínio do tempo para um sinal unidimensional
    """

    return {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'variance': float(np.var(signal)),
        'min': float(np.min(signal)),
        'max': float(np.max(signal)),
        'range': float(np.ptp(signal)),
        'rms': float(np.sqrt(np.mean(signal ** 2))),
        'skewness': float(skew(signal)),
        'kurtosis': float(kurtosis(signal)),
        'iqr': float(np.percentile(signal, 75) - np.percentile(signal, 25)),
        'zero_crossings': int(np.sum(np.diff(np.signbit(signal)))),
        'num_peaks': int(len(find_peaks(signal, distance=50)[0]))
    }

def extract_frequency_features(signal: np.ndarray, fs: float = 100.0) -> Dict[str, float]:

    """
    Extrai características espectrais do sinal
    """

    N = len(signal)
    fft_vals = fft(signal)
    mag = np.abs(fft_vals)[:N // 2]
    freqs = fftfreq(N, 1/fs)[:N // 2]

    mag_sum = np.sum(mag)
    centroid = np.sum(freqs * mag) / mag_sum if mag_sum > 0 else 0
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum) if mag_sum > 0 else 0
    cumulative = np.cumsum(mag)
    rolloff = freqs[np.searchsorted(cumulative, 0.85 * cumulative[-1])] if cumulative[-1] > 0 else 0
    flux = np.sum(np.diff(mag) ** 2)
    dominant = freqs[np.argmax(mag)]

    return {
        'spectral_centroid': float(centroid),
        'spectral_bandwidth': float(bandwidth),
        'spectral_rolloff': float(rolloff),
        'spectral_flux': float(flux),
        'dominant_frequency': float(dominant),
        'fft_mean': float(mag.mean()),
        'fft_std': float(mag.std()),
        'band_energy_5_15Hz': float(np.sum(mag[(freqs >= 5) & (freqs <= 15)] ** 2))
    }

def extract_shannon_entropy(signal: np.ndarray, bins: int = 100) -> Dict[str, float]:

    """
    Calcula a entropia de Shannon
    """

    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist += 1e-12  # evitar log(0)
    ent = scipy_entropy(hist, base=2)

    return {'shannon_entropy': float(ent)}


# ------------------------------ #
# 3- Pipeline de extração completa
# ------------------------------ #

def pipeline_extract_features(filepath_npz: str) -> Tuple[Dict[str, float], dict]:

    """
    Pipeline completo para extrair características de um arquivo .npz processado
    """

    signal, metadata = load_signal_processado(filepath_npz)
    fs = metadata.get('fs', 100)

    if len(signal.shape) == 2:
        signal = signal[:, 0] 

    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, fs))
    features.update(extract_shannon_entropy(signal))

    return features, metadata


# ----------------------------- #
# 4- Salvar características extraídas
# ----------------------------- #

def save_features(features: Dict[str, float], metadata: dict,
                  output_dir: str = "../data/features") -> str:
    
    """
    Salva as features extraídas em um arquivo JSON estruturado
    """

    ecg_id = metadata.get('ecg_id', -1)
    timestamp = metadata.get('timestamp', 'unknown')
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{ecg_id:05d}_features.json")

    estrutura = {
        'ecg_id': ecg_id,
        'timestamp_processado': timestamp,
        'fs': metadata.get('fs'),
        'features': features
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(estrutura, f, indent=2, ensure_ascii=False)

    return filename


# ----------------------------- #
# 6- Visualização de features
# ----------------------------- #

def visualizar_features(features_dict: dict) -> None:

    """
    Exibe um gráfico de barras com as features extraídas
    """

    if "features" in features_dict:
        features_dict = features_dict["features"]

    nomes = list(features_dict.keys())
    valores = list(features_dict.values())

    plt.figure(figsize=(12, 6))
    plt.bar(nomes, valores, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Valor")
    plt.title("Features Extraídas")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# -------------------------- #
# 5- Exemplo de uso (manual)
# -------------------------- #

if __name__ == '__main__':

    arquivo_npz = "../data/processed/records000/00001_processed.npz"

    features, metadata = pipeline_extract_features(arquivo_npz)
    
    caminho_saida = save_features(features, metadata)

    print(f"Features salvas em: {caminho_saida}")

    visualizar_features({'features': features})
