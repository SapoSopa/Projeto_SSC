# Módulo de pré-processamento de sinais
# Contém funções para limpeza, filtragem e preparação dos dados

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from typing import Tuple, Optional, Union
import warnings

# Carregar dados WFDB compatível com PTB-XL dataset;
def load_signal_data(filepath: str) -> Tuple[np.ndarray, dict]:
    try:
        record = wfdb.rdrecord(filepath)        # Carrega registro WFDB;
        metadata = {
            'fs': record.fs,                    # Frequência de amostragem;
            'sig_name': record.sig_name,        # Nomes das derivações;
            'n_samples': len(record.p_signal),  # Número de amostras;
            'n_channels': len(record.sig_name)  # Número de canais;
        }
        return record.p_signal, metadata
    except Exception as e:
        raise ValueError(f"Erro ao carregar arquivo {filepath}: {str(e)}")

# Aplicar filtros digitais Butterworth ao sinal;
def aplicar_filtro(sinal: np.ndarray, fs: int, tipo: str = 'bandpass', 
                   frequencias: Tuple[float, float] = (0.5, 45.0), ordem: int = 4) -> np.ndarray:
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)

    nyquist = 0.5 * fs                          # Frequência de Nyquist;

    if tipo == 'bandpass':
        if len(frequencias) != 2:
            raise ValueError("Para filtro bandpass, forneça duas frequências (baixa, alta)")
        freq_normalizada = [f / nyquist for f in frequencias]
    elif tipo in ['lowpass', 'highpass']:
        freq_normalizada = frequencias[0] / nyquist
    else:
        raise ValueError("Tipo de filtro deve ser 'bandpass', 'lowpass' ou 'highpass'")

    # Verificação crítica de Nyquist para evitar instabilidade;
    if np.any(np.array(freq_normalizada) >= 1.0):
        warnings.warn("Frequência de corte próxima ou superior à frequência de Nyquist")
    
    b, a = signal.butter(ordem, freq_normalizada, btype=tipo)
    
    sinal_filtrado = np.zeros_like(sinal)
    
    # Filtragem de fase zero para preservar forma de onda;
    for i in range(sinal.shape[1]):
        sinal_filtrado[:, i] = signal.filtfilt(b, a, sinal[:, i])
    
    return sinal_filtrado

# Normalizar sinal usando métodos estatísticos como z-score, min-max e robust;
def normalizar_sinal(sinal: np.ndarray, metodo: str = 'zscore') -> np.ndarray:
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)
    
    sinal_normalizado = np.zeros_like(sinal)

    for i in range(sinal.shape[1]):
        if metodo == 'minmax':
            minimo = np.min(sinal[:, i])
            maximo = np.max(sinal[:, i])
            if maximo - minimo == 0:            # Proteção contra divisão por zero, implicada por sinal constante;
                sinal_normalizado[:, i] = sinal[:, i]
            else:
                sinal_normalizado[:, i] = (sinal[:, i] - minimo) / (maximo - minimo)
                
        elif metodo == 'zscore':
            media = np.mean(sinal[:, i])
            desvio = np.std(sinal[:, i])
            if desvio == 0:                     # Proteção contra divisão por zero, implicada por desvio zero;
                sinal_normalizado[:, i] = sinal[:, i] - media
            else:
                sinal_normalizado[:, i] = (sinal[:, i] - media) / desvio
                
        elif metodo == 'robust':
            mediana = np.median(sinal[:, i])
            mad = np.median(np.abs(sinal[:, i] - mediana))
            if mad == 0:                        # Proteção contra divisão por zero, implicada por Mediana Absoluta de Desvio (MAD) zero;
                sinal_normalizado[:, i] = sinal[:, i] - mediana
            else:
                sinal_normalizado[:, i] = (sinal[:, i] - mediana) / mad
        else:
            raise ValueError("Método inválido: use 'zscore', 'minmax' ou 'robust'")
        
    return sinal_normalizado

# Remover desvio da linha de base usando filtro passa-alta;
def remover_baseline_drift(sinal: np.ndarray, fs: int, freq_corte: float = 0.5) -> np.ndarray:
    return aplicar_filtro(sinal, fs, tipo='highpass', frequencias=(freq_corte,))

# Detectar outliers usando z-score;
def detectar_outliers(sinal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)
    
    outliers = np.zeros_like(sinal, dtype=bool)
    for i in range(sinal.shape[1]):             # Z-score para cada canal;
        z_scores = np.abs((sinal[:, i] - np.mean(sinal[:, i])) / np.std(sinal[:, i]))
        outliers[:, i] = z_scores > threshold
    
    return outliers

# Aplicar janelamento ao sinal para análise espectral;
def aplicar_janelamento(sinal: np.ndarray, tipo_janela: str = 'hann') -> np.ndarray:
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)
    
    n_samples = sinal.shape[0]
    
    if tipo_janela == 'hann':
        janela = signal.windows.hann(n_samples)
    elif tipo_janela == 'hamming':
        janela = signal.windows.hamming(n_samples)
    elif tipo_janela == 'blackman':
        janela = signal.windows.blackman(n_samples)
    elif tipo_janela == 'kaiser':
        janela = signal.windows.kaiser(n_samples, beta=8.6)
    else:
        raise ValueError("Tipo de janela inválido")
    
    sinal_janelado = sinal.copy()
    for i in range(sinal.shape[1]):             # Aplicar janela a cada canal;
        sinal_janelado[:, i] = sinal[:, i] * janela
    
    return sinal_janelado

# Verificar qualidade do sinal com métricas básicas;
def verificar_qualidade_sinal(sinal: np.ndarray, fs: int) -> dict:
    if len(sinal.shape) == 1:
        sinal = sinal.reshape(-1, 1)
    
    metricas = {}
    
    for i in range(sinal.shape[1]):
        canal = sinal[:, i]
        nome_canal = f'canal_{i}'
        
        metricas[nome_canal] = {
            'snr_estimado': 20 * np.log10(np.std(canal) / (np.std(np.diff(canal)) + 1e-10)),
            'amplitude_maxima': np.max(np.abs(canal)),
            'saturacao': np.sum(np.abs(canal) > 0.95 * np.max(np.abs(canal))) / len(canal),
            'zero_crossings': np.sum(np.diff(np.sign(canal)) != 0),
            'rms': np.sqrt(np.mean(canal**2))
        }
    
    return metricas

# Pipeline completo de pré-processamento
def pipeline_preprocessamento(filepath: str, 
                            aplicar_filtro_flag: bool = True,
                            normalizar_flag: bool = True,
                            remover_deriva: bool = True) -> Tuple[np.ndarray, dict]:
    # Carregar dados
    sinal, metadata = load_signal_data(filepath)
    fs = metadata['fs']
    
    # Sequência padrão de processamento
    if remover_deriva:
        sinal = remover_baseline_drift(sinal, fs)
    
    if aplicar_filtro_flag:
        sinal = aplicar_filtro(sinal, fs, tipo='bandpass', frequencias=(0.5, 45.0))
    
    if normalizar_flag:
        sinal = normalizar_sinal(sinal, metodo='zscore')
    
    # Avaliar qualidade final
    qualidade = verificar_qualidade_sinal(sinal, fs)
    metadata['qualidade'] = qualidade
    
    return sinal, metadata

# Salvar dados processados de forma organizada
def salvar_dados_processados(sinal: np.ndarray, metadata: dict, 
                           ecg_id: int, 
                           output_dir: str = "../data/processed") -> Tuple[str, str]:
    # Calcular pasta baseada no ECG ID (grupos de 1000)
    folder_number = (ecg_id - 1) // 1000
    folder_name = f"records{folder_number:03d}"
    
    # Criar diretório hierárquico
    pasta_destino = os.path.join(output_dir, folder_name)
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Timestamp para controle de versão
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Nomes dos arquivos com ID formatado
    arquivo_sinal = os.path.join(pasta_destino, f"{ecg_id:05d}_processed.npz")
    arquivo_metadata = os.path.join(pasta_destino, f"{ecg_id:05d}_metadata.json")
    
    # Salvar sinal processado (formato compactado NPZ)
    np.savez_compressed(arquivo_sinal, 
                       sinal=sinal,
                       ecg_id=ecg_id,
                       timestamp=timestamp,
                       shape=sinal.shape,
                       fs=metadata.get('fs', 100))
    
    # Preparar metadados completos para salvamento
    metadata_completo = {
        'processamento': {
            'timestamp': timestamp,
            'ecg_id': ecg_id,
            'pasta_destino': folder_name,
            'versao_preprocessing': '1.0'
        },
        'dados_originais': {
            'fs': metadata.get('fs'),
            'sig_name': metadata.get('sig_name'),
            'n_samples': metadata.get('n_samples'),
            'n_channels': metadata.get('n_channels'),
            'duracao_segundos': metadata.get('n_samples', 0) / metadata.get('fs', 1)
        },
        'qualidade': metadata.get('qualidade', {}),
        'estatisticas': {}
    }
    
    # Adicionar estatísticas do sinal processado
    if len(sinal.shape) == 2:
        metadata_completo['estatisticas'] = {
            'amplitude_media_global': float(np.mean(sinal)),
            'amplitude_std_global': float(np.std(sinal)),
            'amplitude_min_global': float(np.min(sinal)),
            'amplitude_max_global': float(np.max(sinal)),
            'amplitude_rms_global': float(np.sqrt(np.mean(sinal**2))),
            'canais_com_boa_qualidade': 0
        }
        
        # Contar canais com boa qualidade
        if 'qualidade' in metadata:
            canais_bons = sum(1 for i in range(metadata.get('n_channels', 0))
                            if f'canal_{i}' in metadata['qualidade'] and 
                            metadata['qualidade'][f'canal_{i}'].get('snr_estimado', 0) >= 15)
            metadata_completo['estatisticas']['canais_com_boa_qualidade'] = canais_bons
    
    # Salvar metadados (formato JSON estruturado)
    with open(arquivo_metadata, 'w', encoding='utf-8') as f:
        json.dump(metadata_completo, f, indent=2, default=str, ensure_ascii=False)
    
    return arquivo_sinal, arquivo_metadata