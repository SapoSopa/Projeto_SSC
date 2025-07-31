"""
Módulo de extração de características (features)
Contém funções para extrair features de sinais temporais e espectrais
"""
# Criado no chat
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List

def extract_time_domain_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extrai características no domínio do tempo
    
    Args:
        data: Sinal de entrada
        
    Returns:
        Dicionário com as características extraídas
    """
    features = {
        'mean': np.mean(data),
        'std': np.std(data),
        'variance': np.var(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.max(data) - np.min(data),
        'rms': np.sqrt(np.mean(data**2)),
        'skewness': skew(data),
        'kurtosis': kurtosis(data),
        'zero_crossings': len(np.where(np.diff(np.signbit(data)))[0])
    }
    return features

def extract_frequency_domain_features(data: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Extrai características no domínio da frequência
    
    Args:
        data: Sinal de entrada
        fs: Frequência de amostragem
        
    Returns:
        Dicionário com as características espectrais
    """
    # Calcular FFT
    fft = np.fft.fft(data)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(data), 1/fs)
    
    # Considerar apenas metade positiva do espectro
    n = len(data) // 2
    magnitude = magnitude[:n]
    freqs = freqs[:n]
    
    features = {
        'spectral_centroid': np.sum(freqs * magnitude) / np.sum(magnitude),
        'spectral_bandwidth': np.sqrt(np.sum(((freqs - features.get('spectral_centroid', 0))**2) * magnitude) / np.sum(magnitude)),
        'spectral_rolloff': freqs[np.where(np.cumsum(magnitude) >= 0.85 * np.sum(magnitude))[0][0]],
        'spectral_flux': np.sum(np.diff(magnitude)**2),
        'dominant_frequency': freqs[np.argmax(magnitude)]
    }
    return features

def extract_all_features(data: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Extrai todas as características disponíveis
    
    Args:
        data: Sinal de entrada
        fs: Frequência de amostragem
        
    Returns:
        Dicionário com todas as características
    """
    time_features = extract_time_domain_features(data)
    freq_features = extract_frequency_domain_features(data, fs)
    
    all_features = {**time_features, **freq_features}
    return all_features
