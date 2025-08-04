# Módulo Técnico: Extração de Características de Sinais Fisiológicos (`extract_features.py`)

Este módulo realiza a extração automatizada de **características (features)** de sinais fisiológicos (ex: ECG) para posterior análise, visualização ou uso em algoritmos de aprendizado de máquina.

---

## 📊 Classificação por Necessidade

**🔴 CRÍTICAS**: `load_signal_processado`, `extract_time_features`, `extract_frequency_features`  
**🟡 IMPORTANTES**: `extract_shannon_entropy`, `pipeline_extract_features`  
**🟢 ÚTEIS**: `save_features`, `visualizar_features`

---

## Funções Principais

### 🔴 
#### ➡️ `load_signal_processado(filepath_npz: str)`

**Descrição**: Carrega sinais `.npz` processados e seus metadados.

**Necessidade**: **CRÍTICA** – Todo pipeline depende dessa função inicial.

**Parâmetros**:
- `filepath_npz`: Caminho para o arquivo `.npz`.

**Retorna**:
- Sinal (np.ndarray)
- Metadados: frequência (`fs`), ID, shape, timestamp

---

#### ➡️ `extract_time_features(signal: np.ndarray)`

**Descrição**: Extrai características estatísticas e morfológicas do domínio do tempo.

**Necessidade**: **CRÍTICA** – Fundamentais para detectar padrões rítmicos e comportamentos anormais do ECG.

| Feature          | Descrição                                                           | Importância Técnica/Clínica                               |
|------------------|----------------------------------------------------------------------|----------------------------------------------------------|
| `mean`           | Média do sinal                                                       | Nível de base ou offset                                  |
| `std`, `variance`| Dispersão/amplitude de oscilação                                     | Indicador de variabilidade do ritmo cardíaco             |
| `min`, `max`     | Valores extremos                                                     | Detecta picos ou quedas acentuadas                       |
| `range`          | max - min                                                            | Medida geral da amplitude                                |
| `rms`            | Raiz média quadrada                                                  | Potência média do sinal                                  |
| `skewness`       | Assimetria                                                           | Pode indicar anormalidades nas ondas                     |
| `kurtosis`       | Curtose                                                              | Identifica presença de picos ou planicidade              |
| `iqr`            | Intervalo interquartil                                               | Versão robusta da variabilidade                          |
| `zero_crossings` | Nº de vezes que o sinal troca de positivo para negativo e vice-versa | Complexidade oscilatória e ruído                         |
| `num_peaks`      | Picos no sinal (ex: batimentos)                                      | Detecta frequência cardíaca e possíveis arritmias        |

---

#### ➡️ `extract_frequency_features(signal: np.ndarray, fs=100.0)`

**Descrição**: Extrai informações espectrais baseadas em FFT.

**Necessidade**: **CRÍTICA** – Essencial para análise de ritmos, HRV e identificação de padrões de frequência.

| Feature                 | Descrição                                      | Importância Técnica/Clínica                                |
|------------------------|-----------------------------------------------|-------------------------------------------------------------|
| `spectral_centroid`    | Frequência média ponderada                    | Frequência dominante geral do sinal                         |
| `spectral_bandwidth`   | Largura de banda                              | Dispersão das frequências                                   |
| `spectral_rolloff`     | Frequência onde 85% da energia está contida   | Usada para identificar concentração de energia              |
| `spectral_flux`        | Variação no espectro entre frames             | Mudanças abruptas – sensível a transientes                  |
| `dominant_frequency`   | Pico espectral principal                      | Frequência predominante (ex: frequência cardíaca)           |
| `fft_mean`, `fft_std`  | Estatísticas básicas da magnitude             | Resumo da energia espectral                                 |
| `band_energy_5_15Hz`   | Energia entre 5 e 15 Hz                       | Banda típica do ECG: ondas P, QRS e T                       |

---

### 🟡
#### ➡️ `extract_shannon_entropy(signal: np.ndarray, bins=100)`

**Descrição**: Mede a **complexidade** do sinal usando entropia de Shannon.

**Necessidade**: **IMPORTANTE** – Útil para diferenciação entre sinais regulares e caóticos.

| Feature           | Descrição                                   | Importância                                  |
|------------------|---------------------------------------------|----------------------------------------------|
| `shannon_entropy`| Entropia da distribuição de amplitudes      | Mais alta: sinais caóticos ou ruidosos       |
|                  |                                             | Mais baixa: sinais periódicos ou regulares   |

---

### ⚙️
#### ➡️ `pipeline_extract_features(filepath_npz: str)`

**Descrição**: Executa a extração completa em um único comando.

**Necessidade**: **IMPORTANTE** – Automatiza carregamento, análise e organização dos dados.

**Etapas**:
1. Carrega sinal com `load_signal_processado`
2. Seleciona canal 0 se necessário
3. Executa:
   - `extract_time_features`
   - `extract_frequency_features`
   - `extract_shannon_entropy`

---

### 🟢
#### ➡️ `save_features(features, metadata, output_dir="../data/features")`

**Descrição**: Salva os dados extraídos em `.json` formatado.

**Necessidade**: **ÚTIL** – Padroniza armazenamento de resultados para uso futuro ou treinamentos de modelos.

**Exemplo de saída**:
```json
{
  "ecg_id": 1,
  "timestamp_processado": "20250803_153022",
  "fs": 100,
  "features": {
    "mean": 0.02,
    "std": 0.31,
    "spectral_centroid": 7.5,
    "shannon_entropy": 4.82,
    ...
  }
}
```

---

#### ➡️ `visualizar_features(features_dict: dict)`

**Descrição**: Exibe um gráfico de barras com as **features extraídas**, facilitando a análise visual dos valores numéricos obtidos.

**Necessidade**: **ÚTIL** – Ideal para inspeção visual das características extraídas, comparação entre registros ou verificação de anomalias.

**Parâmetros**:
- `features_dict`: Um dicionário com as features. Pode ser:
  - O dicionário `features` retornado por `pipeline_extract_features(...)`
  - Ou um dicionário com chave `"features"` contendo os dados numéricos.

**Funcionamento**:
- Utiliza `matplotlib` para gerar um gráfico de barras.
- Cada feature (ex: `mean`, `std`, `shannon_entropy`, etc.) aparece no eixo X.
- O valor correspondente é mostrado no eixo Y.

**Exemplo de uso**:
```python
features, metadata = pipeline_extract_features("exemplo.npz")
visualizar_features({'features': features})
```

---

## 📈 Aplicações das Features

- **Diagnóstico Automático**: arritmias, fibrilação, anomalias espectrais
- **Classificação**: normal vs. patológico, tipos de batimento
- **Análise de HRV**: variabilidade espectral cardíaca
- **Treinamento de Modelos de ML**: entrada vetorial estruturada

---

## 📦 Dependências

```python
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import json, os
import matplotlib.pyplot as plt
```

---

## 🔗 Integração com Módulo de Pré-Processamento

Este módulo deve ser usado **após** o processamento descrito em `preprocessamento_tecnica.md`.  
Ele consome os arquivos `.npz` gerados e produz arquivos `.json` com os vetores de características organizados.

---

## ✅ Conclusão

O `extract_features.py` é uma etapa essencial em sistemas de análise e classificação de sinais ECG, transformando sinais brutos em **representações numéricas interpretáveis**, com forte base estatística, espectral e informacional. Agora, com a função `visualizar_features`, também permite uma visualização gráfica clara e rápida das informações extraídas.
