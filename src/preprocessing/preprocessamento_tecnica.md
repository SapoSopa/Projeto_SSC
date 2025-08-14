# Módulo de Pré-processamento de Sinais ECG

Este módulo contém funções essenciais para o pré-processamento de sinais ECG, especificamente otimizado para trabalhar com o dataset PTB-XL.

## 📊 Classificação por Necessidade

**🔴 CRÍTICAS**: `load_signal_data`, `aplicar_filtro`, `normalizar_sinal`  
**🟡 IMPORTANTES**: `remover_baseline_drift`, `detectar_outliers`, `aplicar_janelamento`  
**🟢 ÚTEIS**: `verificar_qualidade_sinal`, `pipeline_preprocessamento`, `salvar_dados_processados`

---

## Funções Principais

### 🔴 
#### ➡️ `load_signal_data(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]`

**Descrição**: Carrega dados de sinal de arquivos WFDB (formato usado pelo PTB-XL dataset).

**Necessidade**: **CRÍTICA** - PTB-XL usa formato WFDB não suportado nativamente pelo NumPy/Pandas.

**Parâmetros**:
- `filepath`: Caminho para o arquivo de dados (sem extensão .hea/.dat)

**Retorna**:
- Tupla contendo:
  - Array numpy com os dados do sinal (amostras x canais)
  - Dicionário com metadados (fs, sig_name, n_samples, n_channels)

**Exemplo de uso**:
```python
sinal, metadata = load_signal_data('./data/records100/00001_lr')
print(f"Frequência de amostragem: {metadata['fs']} Hz")
print(f"Número de canais: {metadata['n_channels']}")
```

**Observações**:
- Compatível com formato WFDB usado no PTB-XL
- Trata automaticamente erros de carregamento
- Metadados incluem informações essenciais para processamento posterior
- Casting explícito para tipos Python nativos nos metadados

---

#### ➡️ `aplicar_filtro(sinal: np.ndarray, fs: int, tipo: str = 'bandpass', frequencias: Tuple[float, float] = (0.5, 45.0), ordem: int = 4) -> np.ndarray`

**Descrição**: Aplica filtros digitais Butterworth ao sinal ECG para remoção de ruídos e artefatos.

**Necessidade**: **CRÍTICA** - ECG clínico contém ruídos que reduzem acurácia em >30%.

**Ruídos removidos**: Interferência de linha (50/60 Hz), deriva respiratória (0.05-0.5 Hz), ruído muscular (>100 Hz).

**Parâmetros**:
- `sinal`: Array numpy com shape (amostras, canais)
- `fs`: Frequência de amostragem em Hz
- `tipo`: Tipo do filtro ('bandpass', 'lowpass', 'highpass')
- `frequencias`: Tupla com frequências de corte (Hz)
- `ordem`: Ordem do filtro (padrão: 4)

**Retorna**:
- Sinal filtrado com mesmo shape do input

**Validações Implementadas**:
- **Entrada**: Verifica se sinal é array numpy e dimensões válidas
- **Frequência**: Valida fs > 0 e frequências dentro de limites de Nyquist
- **Tipo**: Verifica tipos de filtro suportados
- **Consistência**: Para bandpass, verifica freq_baixa < freq_alta

**Filtros Recomendados para ECG**:
- **Bandpass (0.5-45 Hz)**: Remove deriva da linha de base e ruído de alta frequência
- **Highpass (0.5 Hz)**: Remove apenas deriva da linha de base
- **Lowpass (45 Hz)**: Remove ruído de alta frequência mantendo componentes do ECG

**Exemplo de uso**:
```python
# Filtro passa-banda padrão para ECG
sinal_filtrado = aplicar_filtro(sinal, fs=500, tipo='bandpass', frequencias=(0.5, 45.0))

# Filtro passa-alta para remover deriva
sinal_sem_deriva = aplicar_filtro(sinal, fs=500, tipo='highpass', frequencias=(0.5,))
```

**Observações**:
- Usa `filtfilt` para filtragem de fase zero (preserva morfologia)
- Verifica automaticamente frequências de Nyquist com warnings
- Trata sinais 1D e 2D automaticamente
- 0.5 Hz baixa preserva onda T e segmento ST
- 45 Hz alta remove ruído mantendo QRS (até ~40 Hz)

---

#### ➡️ `normalizar_sinal(sinal: np.ndarray, metodo: str = 'zscore') -> np.ndarray`

**Descrição**: Normaliza o sinal usando diferentes métodos estatísticos para padronizar amplitudes.

**Necessidade**: **CRÍTICA** - Amplitudes de ECG variam 10-100x entre pacientes, impossibilitando convergência de algoritmos ML.

**Parâmetros**:
- `sinal`: Array numpy com shape (amostras, canais)
- `metodo`: Método de normalização ('zscore', 'minmax', 'robust')

**Métodos Disponíveis**:

1. **Min-Max**: `(x - min) / (max - min)`
   - Normaliza para intervalo [0, 1]
   - Sensível a outliers
   - Útil para visualização
   - **Proteção**: Se range = 0, mantém sinal original

2. **Z-Score** (padrão): `(x - μ) / σ`
   - Centraliza em zero com desvio padrão 1
   - Ideal para redes neurais e algoritmos baseados em gradiente
   - Preserva forma da distribuição
   - **Proteção**: Se σ = 0, subtrai apenas a média

3. **Robust**: `(x - mediana) / MAD`
   - Usa mediana e MAD (Median Absolute Deviation)
   - Resistente a outliers
   - Recomendado para dados clínicos com artefatos
   - **Proteção**: Se MAD = 0, subtrai apenas a mediana

**Exemplo de uso**:
```python
# Normalização Z-score (recomendada para ECG)
sinal_norm = normalizar_sinal(sinal, metodo='zscore')

# Normalização robusta para dados com outliers
sinal_robust = normalizar_sinal(sinal, metodo='robust')
```

**Observações**:
- Amplitude varia de 0.1-5.0 mV (50x variação) inter-paciente
- Offset de -2 a +2 mV, inter-paciente
- Impedância varia com eletrodos, pele e idade, inter-paciente
- **Proteções robustas** contra divisão por zero implementadas
- Processamento por canal individual

---

### 🟡
#### ➡️ `remover_baseline_drift(sinal: np.ndarray, fs: int, freq_corte: float = 0.5) -> np.ndarray`

**Descrição**: Remove deriva da linha de base usando filtro passa-alta.

**Necessidade**: **IMPORTANTE** - Deriva afeta análise de segmento ST e pode mascarar arritmias.

**Parâmetros**:
- `sinal`: Array numpy com shape (amostras, canais)
- `fs`: Frequência de amostragem
- `freq_corte`: Frequência de corte para remoção da deriva (Hz)

**Sobre Baseline Drift**:
- Deriva lenta da linha de base devido a respiração, movimento do paciente, polarização de eletrodos ou variação de temperatura
- Frequências típicas: 0.05-0.5 Hz
- Pode afetar análise de segmentos ST e medidas de amplitude

**Implementação**:
- Usa `aplicar_filtro()` internamente com tipo='highpass'
- Filtragem de fase zero para preservar morfologia
- Aplicado canal por canal automaticamente

**Exemplo de uso**:
```python
# Remoção padrão de deriva (0.5 Hz)
sinal_sem_deriva = remover_baseline_drift(sinal, fs=500)

# Remoção mais agressiva
sinal_limpo = remover_baseline_drift(sinal, fs=500, freq_corte=1.0)
```

---

#### ➡️ `detectar_outliers(sinal: np.ndarray, threshold: float = 3.0) -> np.ndarray`

**Descrição**: Detecta outliers usando método z-score para identificação de artefatos.

**Necessidade**: **IMPORTANTE** - PTB-XL contém artefatos clínicos que causam overfitting em modelos.

**Tipos detectados**: Artefatos de movimento, falhas de aquisição, interferência de equipamentos.

**Parâmetros**:
- `sinal`: Array numpy com shape (amostras, canais)
- `threshold`: Limite do z-score para considerar outlier (padrão: 3.0)

**Retorna**:
- Array booleano indicando posições dos outliers

**Proteções Implementadas**:
- **Sinal constante**: Se σ = 0, nenhum outlier é detectado
- **Processamento por canal**: Cada canal analisado independentemente
- **Validação de entrada**: Conversão automática para 2D se necessário

**Interpretação dos Thresholds**:
- `threshold=2.0`: ~5% dos dados removidos (mais sensível)
- `threshold=3.0`: ~0.3% dos dados removidos (padrão)
- `threshold=4.0`: ~0.01% dos dados removidos (menos sensível)

**Exemplo de uso**:
```python
outliers = detectar_outliers(sinal, threshold=3.0)
print(f"Outliers detectados: {np.sum(outliers)} de {len(sinal)} amostras")

# Visualizar outliers
import matplotlib.pyplot as plt
plt.plot(sinal)
plt.scatter(np.where(outliers), sinal[outliers], color='red', s=20)
```

---

### 🟢 
#### ➡️ `verificar_qualidade_sinal(sinal: np.ndarray, fs: int) -> Dict[str, Dict[str, float]]`

**Descrição**: Calcula métricas de qualidade do sinal para avaliação automática.

**Necessidade**: **ÚTIL** - Controle automático de qualidade para sistemas de produção e debug.

**Métricas Calculadas**:

1. **SNR Estimado**: `20 * log10(std(sinal) / (std(diff(sinal)) + 1e-10))`
   - **Interpretação**: >20 dB = boa qualidade, <10 dB = problemático
   - **Baseado em**: Diferença entre variabilidade do sinal vs. ruído
   - **Proteção**: Adição de 1e-10 para evitar divisão por zero

2. **Amplitude Máxima**: `max(abs(sinal))`
   - **Normal ECG**: 0.5-3.0 mV
   - **Problema**: >10 mV (saturação), <0.1 mV (ganho baixo)

3. **Saturação**: `% amostras > 95% do máximo`
   - **Normal**: <1%
   - **Problema**: >5% indica saturação do amplificador
   - **Proteção**: Se amplitude_max = 0, saturação = 0.0

4. **Zero Crossings**: `Número de cruzamentos por zero`
   - **Relacionado**: Conteúdo de frequência
   - **Anormal**: Muito baixo (deriva) ou muito alto (ruído)

5. **RMS**: `sqrt(mean(sinal²))`
   - **Energia**: Medida da potência média do sinal
   - **Útil**: Comparação entre registros

**Proteções Implementadas**:
- **Sinais constantes**: Tratamento especial para evitar divisões por zero
- **Valores extremos**: Proteções contra overflow e underflow
- **Casting seguro**: Conversão explícita para float

**Exemplo de uso**:
```python
qualidade = verificar_qualidade_sinal(sinal, fs=500)
for canal, metricas in qualidade.items():
    if metricas['snr_estimado'] < 15:
        print(f"⚠️ {canal}: SNR baixo ({metricas['snr_estimado']:.1f} dB)")
```

---

#### ➡️ `pipeline_preprocessamento(filepath: str, aplicar_filtro_flag: bool = True, normalizar_flag: bool = True, remover_deriva: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]`

**Descrição**: Pipeline completo de pré-processamento com todas as etapas integradas.

**Necessidade**: **CONVENIENTE** - Garante reprodutibilidade e sequência otimizada de processamento.

**Parâmetros**:
- `filepath`: Caminho para arquivo de dados
- `aplicar_filtro_flag`: Se deve aplicar filtro passa-banda
- `normalizar_flag`: Se deve normalizar o sinal
- `remover_deriva`: Se deve remover deriva da linha de base

**Pipeline Padrão** (ordem otimizada):
1. Carregamento dos dados WFDB
2. **Validação inicial**: Verifica se sinal tem pelo menos 100 amostras (1s a 100Hz)
3. Remoção de deriva da linha de base (0.5 Hz highpass) - se `remover_deriva=True`
4. Filtragem passa-banda (0.5-45 Hz) - se `aplicar_filtro_flag=True`
5. Normalização Z-score - se `normalizar_flag=True`
6. Verificação de qualidade final

**POR QUE ESTA ORDEM**:
1. **Deriva primeiro**: Remove componentes de baixa frequência que afetam filtros
2. **Filtro depois**: Opera em sinal com linha de base estável
3. **Normalização por último**: Aplica em sinal já limpo

**Validações e Proteções**:
- **Sinal muito curto**: Warning para sinais < 100 amostras
- **Tratamento de erros**: RuntimeError com mensagem informativa
- **Qualidade integrada**: Metadados incluem métricas de qualidade

**Exemplo de uso**:
```python
# Pipeline completo
sinal_processado, metadata = pipeline_preprocessamento('./data/00001_lr')

# Pipeline personalizado
sinal_custom, metadata = pipeline_preprocessamento(
    './data/00001_lr',
    aplicar_filtro_flag=True,
    normalizar_flag=False,  # Sem normalização
    remover_deriva=True
)
```

---

#### ➡️ `salvar_dados_processados(sinal: np.ndarray, metadata: Dict[str, Any], ecg_id: int, output_dir: str = "../data/processed") -> Tuple[str, str]`

**Descrição**: Salva dados processados em estrutura hierárquica organizacional com controle de versão e rastreabilidade completa.

**Necessidade**: **CONVENIENTE** - Padroniza salvamento hierárquico escalável para grandes datasets como PTB-XL.

**Parâmetros**:
- `sinal`: Array numpy com sinal processado (n_samples, n_channels)
- `metadata`: Dicionário com metadados completos do processamento
- `ecg_id`: Identificador numérico único do ECG (usado para organização hierárquica)
- `output_dir`: Diretório base de destino (padrão: "../data/processed")

**Validações de Entrada Implementadas**:
- **Tipo do sinal**: Verifica se é array numpy
- **ECG ID**: Deve ser positivo (>= 1)
- **Sinal vazio**: Verifica se sinal não está vazio
- **Metadados**: Verifica se é dicionário válido

**Retorna**:
- Tupla com caminhos dos arquivos salvos: `(arquivo_sinal, arquivo_metadata)`

**🗂️ Estrutura Hierárquica Criada**:
```
../data/processed/
├── records000/              # ECGs 00001-01000
│   ├── 00001_processed.npz
│   ├── 00001_metadata.json
│   ├── 00002_processed.npz
│   ├── 00002_metadata.json
│   └── ...
├── records001/              # ECGs 01001-02000
│   ├── 01001_processed.npz
│   ├── 01001_metadata.json
│   └── ...
├── records002/              # ECGs 02001-03000
└── relatorio_*.csv          # Relatórios na pasta raiz
```

**📁 Arquivos Gerados por ECG**:

1. **`{ecg_id:05d}_processed.npz`**: Sinal em formato NPZ compactado
   - Contém: `sinal`, `ecg_id`, `timestamp`, `shape`, `fs`
   - Carregamento ultra-rápido com `np.load()`
   - Compressão automática para economia de espaço

2. **`{ecg_id:05d}_metadata.json`**: Metadados estruturados completos
   - **`processamento`**: Timestamp, ID, pasta, versão
   - **`dados_originais`**: Frequência, canais, amostras, duração
   - **`qualidade`**: Métricas por canal (SNR, amplitude, saturação)
   - **`estatisticas`**: Estatísticas globais do sinal processado

**Estatísticas Globais Calculadas**:
- Amplitude média, std, min, max, RMS global
- **Contagem de canais com boa qualidade** (SNR >= 15 dB)
- Casting seguro para tipos JSON-compatíveis

**🔍 Exemplo de Conteúdo dos Arquivos**:

**NPZ (carregamento)**:
```python
data = np.load('00001_processed.npz')
sinal = data['sinal']          # Array (n_samples, n_channels)
ecg_id = data['ecg_id']        # ID numérico
timestamp = data['timestamp']  # Quando foi processado
fs = data['fs']               # Frequência de amostragem
```

**JSON (metadados estruturados)**:
```json
{
  "processamento": {
    "timestamp": "20250130_143022",
    "ecg_id": 1,
    "pasta_destino": "records000",
    "versao_preprocessing": "1.0"
  },
  "dados_originais": {
    "fs": 100,
    "sig_name": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "n_samples": 1000,
    "n_channels": 12,
    "duracao_segundos": 10.0
  },
  "qualidade": {
    "canal_0": {
      "snr_estimado": 18.5,
      "amplitude_maxima": 1.245,
      "saturacao": 0.02,
      "rms": 0.156
    }
  },
  "estatisticas": {
    "amplitude_media_global": 0.001,
    "amplitude_std_global": 0.234,
    "amplitude_min_global": -1.234,
    "amplitude_max_global": 1.456,
    "amplitude_rms_global": 0.345,
    "canais_com_boa_qualidade": 11
  }
}
```

**💡 Exemplo de uso**:
```python
# Processamento individual
sinal_processado, metadata = pipeline_preprocessamento(filepath)
arquivos = salvar_dados_processados(
    sinal_processado, 
    metadata, 
    ecg_id=1,  # ID numérico do ECG
    output_dir="../data/processed"
)

# Carregamento posterior
data = np.load(arquivos[0])
sinal_carregado = data['sinal']

import json
with open(arquivos[1], 'r', encoding='utf-8') as f:
    metadata_carregado = json.load(f)
```

**🚀 Vantagens da Estrutura Hierárquica**:
- **Performance**: Máximo 1000 arquivos por pasta (otimização do filesystem)
- **Escalabilidade**: Suporta datasets grandes como PTB-XL (21K+ registros)
- **Organização**: Estrutura similar aos dados raw para navegação intuitiva
- **Busca Rápida**: ID numérico permite localização direta da pasta
- **Compatibilidade**: Padrão usado em datasets médicos
- **Rastreabilidade**: Timestamping automático e controle de versão

**📊 Fórmula da Organização**:
```python
folder_number = (ecg_id - 1) // 1000
folder_name = f"records{folder_number:03d}"

# Exemplos:
# ECG 1     → records000/
# ECG 1000  → records000/
# ECG 1001  → records001/
# ECG 2500  → records002/
```

---

## ⚙️ Configurações por Contexto

### Mínima (Proof of Concept)
Uso: testes iniciais, protótipos rápidos, validação do código.
```python
sinal, metadata = load_signal_data(filepath)
sinal = aplicar_filtro(sinal, metadata['fs'])
sinal = normalizar_sinal(sinal)

# Salvamento simples
arquivos = salvar_dados_processados(sinal, metadata, ecg_id=1)
```
- Objetivo: carregar, filtrar o ruído e normalizar. Simples e direto.
- Sem verificação de qualidade, sem remoção de outlier, sem deriva de linha de base.
- Útil para ver se a estrutura geral do código funciona.

### Robusta (Pesquisa)
Uso: experimentos com controle maior de qualidade, análise exploratória em papers ou testes comparativos.
```python
sinal, metadata = pipeline_preprocessamento(
    filepath,
    aplicar_filtro_flag=True,
    normalizar_flag=True, 
    remover_deriva=True
)
outliers = detectar_outliers(sinal)

# Salvamento com rastreabilidade
arquivos = salvar_dados_processados(sinal, metadata, ecg_id=experiment_id)
```
- Usa um pipeline mais completo e parametrizado.
- Inclui: Filtro, Normalização, Remoção de baseline drift, Detecção de outliers
- Boa prática para reprodutibilidade e análise científica.

### Produção (Sistema Clínico)
Uso: sistemas usados em ambiente real (ex: hospitais, dispositivos médicos embarcados).
```python
sinal, metadata = pipeline_preprocessamento(filepath)
qualidade = metadata['qualidade']

# Verificação automática de qualidade
canais_ruins = []
for i in range(metadata['n_channels']):
    if qualidade[f'canal_{i}']['snr_estimado'] < 15:
        canais_ruins.append(metadata['sig_name'][i])

if len(canais_ruins) > 3:  # Mais de 3 canais ruins
    print(f"⚠️ Qualidade inadequada: {canais_ruins}")
    # Sinalizar para revisão manual
else:
    # Salvamento hierárquico para produção
    arquivos = salvar_dados_processados(
        sinal, metadata, 
        ecg_id=extrair_id_do_arquivo(filepath),
        output_dir="/dados/processados/producao"
    )
```
- Pipeline automático padronizado com verificação de qualidade
- Rejeição automática de sinais abaixo do padrão
- Estrutura hierárquica para datasets grandes

---

## 📁 Estrutura de Arquivos Gerados

### 🗂️ Organização Hierárquica Completa
```
data/processed/
├── records000/                           # ECGs 00001-01000
│   ├── 00001_processed.npz              # Sinal ECG 1
│   ├── 00001_metadata.json              # Metadados ECG 1
│   ├── 00002_processed.npz              # Sinal ECG 2
│   ├── 00002_metadata.json              # Metadados ECG 2
│   └── ...
├── records001/                           # ECGs 01001-02000
│   ├── 01001_processed.npz
│   ├── 01001_metadata.json
│   └── ...
├── records002/                           # ECGs 02001-03000
├── relatorio_processamento_completo_20250130_143022.csv
├── relatorio_sucessos_20250130_143022.csv
└── relatorio_erros_20250130_143022.csv
```

### 📋 Exemplo de Relatório CSV
```csv
ecg_id,patient_id,age,sex,status,snr_medio_original,snr_medio_final,melhoria_snr,canais_com_boa_qualidade,pasta_destino,arquivo_sinal
1,1,65,0,"✅ Sucesso",16.2,19.8,3.6,12,records000,../data/processed/records000/00001_processed.npz
2,2,45,1,"✅ Sucesso",14.1,18.2,4.1,11,records000,../data/processed/records000/00002_processed.npz
```

---

## 📋 Parâmetros Recomendados para ECG

### Filtragem
- **Passa-banda**: 0.5-45 Hz (padrão AHA/ACC para ECG)
- **Passa-alta**: 0.05-0.5 Hz (remoção de deriva)
- **Passa-baixa**: 40-100 Hz (anti-aliasing)

### Normalização
- **Z-score**: Para análise de machine learning
- **Robust**: Para dados com artefatos
- **Min-max**: Para visualização

### Qualidade do Sinal
- **SNR mínimo**: 15-20 dB para análise automática
- **Saturação máxima**: < 1% das amostras
- **Frequência de amostragem**: 100-500 Hz (PTB-XL: 100/500 Hz)

### Organização de Arquivos
- **Máximo por pasta**: 1000 arquivos (otimização filesystem)
- **Formato de dados**: NPZ compactado (velocidade + economia)
- **Metadados**: JSON estruturado (legibilidade + compatibilidade)

---

## 🔗 Integração com PTB-XL Dataset

Este módulo foi otimizado para trabalhar com:
- Arquivos WFDB (.hea/.dat)
- Frequência de amostragem de 100/500 Hz
- 12 derivações padrão (I, II, III, aVR, aVL, aVF, V1-V6)
- 21,837 registros de 10 segundos
- Estrutura hierárquica escalável para grandes volumes

---

## 📦 Dependências

```python
import numpy as np                           # >= 1.19.0
import pandas as pd                          # >= 1.3.0  
import wfdb                                  # >= 3.4.0 (essencial para PTB-XL)
from scipy import signal                     # >= 1.7.0
from typing import Tuple, Optional, Union, Dict, Any
import json                                  # Biblioteca padrão
import os                                    # Biblioteca padrão
from datetime import datetime                # Biblioteca padrão
import warnings
```
---