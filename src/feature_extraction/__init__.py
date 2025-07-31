"""
feature_extraction.py
Módulo único com todas as funções de extração de características
para sinais temporais, espectrais e de entropia.
"""
import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from typing import Dict


def extract_time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extrai características no domínio do tempo.
    """
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'variance': np.var(signal),
        'min': np.min(signal),
        'max': np.max(signal),
        'range': np.ptp(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'skewness': skew(signal),
        'kurtosis': kurtosis(signal),
        'iqr': np.percentile(signal, 75) - np.percentile(signal, 25),
        'zero_crossings': len(np.where(np.diff(np.signbit(signal)))[0]),
        'num_peaks': len(find_peaks(signal, distance=50)[0]),
        'duration_sec': len(signal) / 500  # fs padrão = 500 Hz
    }


def extract_frequency_domain_features(signal: np.ndarray, fs: float = 1.0) -> Dict[str, float]:
    """
    Extrai características no domínio da frequência.
    """
    N = len(signal)
    fft_vals = fft(signal)
    mag = np.abs(fft_vals)[:N // 2]
    freqs = fftfreq(N, 1/fs)[:N // 2]

    centroid = np.sum(freqs * mag) / np.sum(mag)
    bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * mag) / np.sum(mag))
    cumulative = np.cumsum(mag)
    rolloff = freqs[np.searchsorted(cumulative, 0.85 * cumulative[-1])]
    flux = np.sum(np.diff(mag)**2)
    dominant = freqs[np.argmax(mag)]

    return {
        'spectral_centroid': centroid,
        'spectral_bandwidth': bandwidth,
        'spectral_rolloff': rolloff,
        'spectral_flux': flux,
        'dominant_frequency': dominant,
        'fft_mean': mag.mean(),
        'fft_std': mag.std(),
        'band_energy_5_15Hz': np.sum(mag[(freqs >= 5) & (freqs <= 15)]**2)
    }


def shannon_entropy(signal: np.ndarray, bins: int = 100) -> float:
    """
    Calcula a entropia de Shannon de um vetor 1D.
    """
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist += 1e-12
    return scipy_entropy(hist, base=2)


def spectral_entropy(signal: np.ndarray, fs: float, n_fft: int = 1024) -> float:
    """
    Calcula a entropia espectral usando FFT.
    """
    ps = np.abs(fft(signal, n=n_fft))**2
    ps = ps[:n_fft // 2]
    p_norm = ps / np.sum(ps)
    p_norm += 1e-12
    return -np.sum(p_norm * np.log2(p_norm))


def extract_all_features(signal: np.ndarray, fs: float = 500.0) -> Dict[str, float]:
    """
    Extrai todas as características de tempo, frequência e entropia.
    """
    time_feats = extract_time_domain_features(signal)
    freq_feats = extract_frequency_domain_features(signal, fs)
    entropies = {
        'shannon_entropy': shannon_entropy(signal),
        'spectral_entropy': spectral_entropy(signal, fs)
    }
    return {**time_feats, **freq_feats, **entropies}


# Exemplo de uso
if __name__ == '__main__':
    fs = 500  # Hz
    t = np.linspace(0, 10, fs*10, endpoint=False)
    signal = 0.5 * np.sin(2*np.pi*1.2*t) + 0.05 * np.random.randn(len(t))
    features = extract_all_features(signal, fs)
    for k, v in features.items():
        print(f"{k}: {v:.4f}")
