import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt 
from typing import Dict, Tuple
from typing import Tuple, Optional, Union
import warnings

# Carregar sinal processado de arquivo NPZ;
def load_signal_processado(filepath_npz: str) -> Tuple[np.ndarray, dict]:

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

# Extrair características estatísticas do domínio do tempo;
def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
    
    if len(signal.shape) != 1:
        raise ValueError("Sinal deve ser unidimensional")
    
    # Detectar picos com parâmetros otimizados para ECG
    peaks, _ = find_peaks(signal, distance=50, height=np.std(signal) * 0.5)
    
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
        'num_peaks': int(len(peaks))
    }

# Extrair características espectrais usando FFT;
def extract_frequency_features(signal: np.ndarray, fs: float = 100.0, aplicar_janela: bool = True) -> Dict[str, float]:
   
    if len(signal) == 0:
        return {}
    
    # Garantir que é 1D
    if len(signal.shape) > 1:
        signal = signal.flatten()
    
    N = len(signal)
    
    # Aplicar janelamento se necessário
    if aplicar_janela:
        sinal_janelado = aplicar_janelamento(signal.reshape(-1, 1), tipo_janela='hann')
        sinal_para_fft = sinal_janelado.flatten()
    else:
        sinal_para_fft = signal
    
    # Calcular FFT usando o sinal correto
    fft_vals = fft(sinal_para_fft)
    mag = np.abs(fft_vals)[:N // 2]
    freqs = fftfreq(N, 1/fs)[:N // 2]
    
    # Power spectrum
    power = mag ** 2
    
    # Evitar divisão por zero
    if np.sum(power) == 0:
        return {
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_rolloff': 0.0,
            'spectral_flux': 0.0,
            'dominant_frequency': 0.0,
            'fft_mean': 0.0,
            'fft_std': 0.0,
            'band_energy_0_5_45Hz': 0.0
        }
    
    # Spectral Centroid
    spectral_centroid = np.sum(freqs * power) / np.sum(power)
    
    # Spectral Bandwidth
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * power) / np.sum(power))
    
    # Spectral Rolloff (85% da energia)
    cumsum_power = np.cumsum(power)
    rolloff_point = 0.85 * cumsum_power[-1]
    rolloff_idx = np.where(cumsum_power >= rolloff_point)[0]
    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
    
    # Dominant Frequency
    dominant_idx = np.argmax(power)
    dominant_frequency = freqs[dominant_idx]
    
    # Spectral Flux (variação temporal - aproximação simples)
    spectral_flux = np.sum(np.diff(power) ** 2) if len(power) > 1 else 0.0
    
    # Estatísticas básicas da FFT
    fft_mean = np.mean(mag)
    fft_std = np.std(mag)
    
    # Energia em banda específica (0.5-45 Hz para ECG)
    band_mask = (freqs >= 0.5) & (freqs <= 45.0)
    band_energy = np.sum(power[band_mask]) if np.any(band_mask) else 0.0
    
    return {
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'spectral_flux': float(spectral_flux),
        'dominant_frequency': float(dominant_frequency),
        'fft_mean': float(fft_mean),
        'fft_std': float(fft_std),
        'band_energy_0_5_45Hz': float(band_energy)
    }

# Calcular entropia de Shannon para medir complexidade do sinal;
def extract_shannon_entropy(signal: np.ndarray, bins: int = 100) -> Dict[str, float]:
    
    if len(signal.shape) != 1:
        raise ValueError("Sinal deve ser unidimensional")
    
    # Calcular histograma
    hist, _ = np.histogram(signal, bins=bins, density=False)
    
    # Normalizar para distribuição de probabilidade
    prob_dist = hist / np.sum(hist)
    
    # Evitar log(0) adicionando pequeno valor
    prob_dist = prob_dist + 1e-12
    
    # Calcular entropia usando base 2
    entropy_value = scipy_entropy(prob_dist, base=2)
    
    return {'shannon_entropy': float(entropy_value)}

# Aplicar janelamento ao sinal para análise espectral;
def aplicar_janelamento(sinal: np.ndarray, tipo_janela: str = 'hann') -> np.ndarray:
   
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)
    
    n_samples = sinal.shape[0]
    
    # Selecionar tipo de janela
    if tipo_janela == 'hann':
        janela = signal.windows.hann(n_samples)
    elif tipo_janela == 'hamming':
        janela = signal.windows.hamming(n_samples)
    elif tipo_janela == 'blackman':
        janela = signal.windows.blackman(n_samples)
    elif tipo_janela == 'kaiser':
        janela = signal.windows.kaiser(n_samples, beta=8.6)
    else:
        raise ValueError("Tipo de janela inválido: use 'hann', 'hamming', 'blackman' ou 'kaiser'")
    
    sinal_janelado = sinal.copy()
    for i in range(sinal.shape[1]):
        sinal_janelado[:, i] = sinal[:, i] * janela
    
    return sinal_janelado

# Pipeline completo para extração de características de um arquivo;
def pipeline_extract_features(filepath_npz: str, canal: int = 0) -> Tuple[Dict[str, float], dict]:
    
    # Carregar sinal processado
    signal, metadata = load_signal_processado(filepath_npz)
    fs = metadata.get('fs', 100)
    
    # Selecionar canal apropriado
    if len(signal.shape) == 2:
        if canal >= signal.shape[1]:
            raise ValueError(f"Canal {canal} não existe. Sinal tem {signal.shape[1]} canais.")
        signal = signal[:, canal]
    elif len(signal.shape) == 1:
        if canal != 0:
            raise ValueError("Sinal unidimensional: use canal=0")
    else:
        raise ValueError("Formato de sinal não suportado")
    
    # Extrair todas as características
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, fs))
    features.update(extract_shannon_entropy(signal))
    
    # Adicionar informações do canal aos metadados
    metadata['canal_analisado'] = canal
    
    return features, metadata

# Pipeline para extração rápida de características de um canal específico;
def extract_features_canal(signals: np.ndarray, canal_idx: int, fs: float = 100.0) -> Dict[str, float]:
   
    if len(signals.shape) != 2:
        raise ValueError("Array deve ter shape (amostras, canais)")
    
    if canal_idx >= signals.shape[1]:
        raise ValueError(f"Canal {canal_idx} não existe. Array tem {signals.shape[1]} canais.")
    
    signal = signals[:, canal_idx]
    
    features = {}
    features.update(extract_time_features(signal))
    features.update(extract_frequency_features(signal, fs))
    features.update(extract_shannon_entropy(signal))
    
    return features

# Salvar características extraídas em formato JSON estruturado;
def save_features(features: Dict[str, float], metadata: dict,
                  output_dir: str = "../data/features") -> str:
   
    ecg_id = metadata.get('ecg_id', -1)
    timestamp_processamento = datetime.now().strftime("%Y%m%d_%H%M%S")
    canal = metadata.get('canal_analisado', 0)
    
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"{ecg_id:05d}_canal{canal}_features.json")

    estrutura = {
        'processamento': {
            'ecg_id': ecg_id,
            'canal_analisado': canal,
            'timestamp_extracao': timestamp_processamento,
            'timestamp_preprocessamento': metadata.get('timestamp', 'unknown')
        },
        'dados_originais': {
            'fs': metadata.get('fs', 100),
            'shape': metadata.get('shape', (0, 0))
        },
        'features': features,
        'num_features': len(features)
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(estrutura, f, indent=2, ensure_ascii=False)

    return filename


# Visualizar características extraídas em gráfico de barras;
def visualizar_features(features_dict: dict, titulo: str = "Features Extraídas") -> None:
    
    # Extrair features se estiver em estrutura aninhada
    if "features" in features_dict:
        features = features_dict["features"]
    else:
        features = features_dict
    
    if not features:
        print("Nenhuma feature encontrada para visualizar")
        return
    
    nomes = list(features.keys())
    valores = list(features.values())
    n = len(valores)
    
    # Esquema de cores
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]
    
    # Criar gráfico
    plt.figure(figsize=(14, 8))
    bars = plt.bar(nomes, valores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, valores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{valor:.3e}' if abs(valor) > 1000 or abs(valor) < 0.001 else f'{valor:.3f}',
                ha='center', va='bottom', fontsize=8, rotation=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Valor", fontweight='bold')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

# Pipeline para processar múltiplos canais de um sinal;
def pipeline_multicanal(filepath_npz: str, 
                       canais: list = None,
                       salvar_features: bool = True,
                       output_dir: str = "../data/features") -> Dict[int, Dict[str, float]]:
    
    # Carregar sinal
    signals, metadata = load_signal_processado(filepath_npz)
    fs = metadata.get('fs', 100)
    
    if len(signals.shape) != 2:
        raise ValueError("Sinal deve ser multi-canal (shape: amostras x canais)")
    
    # Definir canais a processar
    if canais is None:
        canais = list(range(signals.shape[1]))
    
    # Processar cada canal
    resultados = {}
    arquivos_salvos = []
    
    for canal in canais:
        try:
            features = extract_features_canal(signals, canal, fs)
            resultados[canal] = features
            
            if salvar_features:
                metadata_canal = metadata.copy()
                metadata_canal['canal_analisado'] = canal
                arquivo = save_features(features, metadata_canal, output_dir)
                arquivos_salvos.append(arquivo)
                
        except Exception as e:
            print(f"⚠️ Erro no canal {canal}: {str(e)}")
            resultados[canal] = {}
    
    if salvar_features and arquivos_salvos:
        print(f"Features salvas para {len(arquivos_salvos)} canais em {output_dir}")
    
    return resultados