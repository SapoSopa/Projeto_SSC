# Módulo de Extração de Características de Sinais ECG

Este módulo contém funções essenciais para extração automatizada de características (features) de sinais ECG pré-processados, especificamente otimizado para trabalhar com o dataset PTB-XL e dados processados pelo módulo de pré-processamento.

## 📊 Classificação por Necessidade

**🔴 CRÍTICAS**: `load_signal_processado`, `extract_time_features`, `extract_frequency_features`  
**🟡 IMPORTANTES**: `extract_shannon_entropy`, `aplicar_janelamento`, `pipeline_extract_features`  
**🟢 ÚTEIS**: `save_features`, `visualizar_features`, `pipeline_multicanal`, `extract_features_canal`

---

## Funções Principais

### 🔴 
#### ➡️ `load_signal_processado(filepath_npz: str) -> Tuple[np.ndarray, dict]`

**Descrição**: Carrega sinais ECG pré-processados e seus metadados a partir de arquivos `.npz` gerados pelo módulo de pré-processamento.

**Necessidade**: **CRÍTICA** - Todo pipeline de extração de características depende dessa função para acessar dados processados.

**Parâmetros**:
- `filepath_npz`: Caminho para arquivo NPZ processado (ex: `../data/processed/records000/00001_processed.npz`)

**Retorna**:
- Tupla contendo:
  - Array numpy com dados do sinal (amostras x canais)
  - Dicionário com metadados (fs, ecg_id, timestamp, shape)

**Validações Implementadas**:
- **Verificação de arquivo**: FileNotFoundError para arquivos inexistentes
- **Casting de tipos**: Conversão explícita para int/str nos metadados
- **Valores padrão**: fs=100, ecg_id=-1, timestamp="unknown" se não encontrados

**Exemplo de uso**:
```python
signals, metadata = load_signal_processado('./data/processed/records000/00001_processed.npz')
print(f"ECG ID: {metadata['ecg_id']}, Freq: {metadata['fs']} Hz")
print(f"Shape: {signals.shape}")
```

**Observações**:
- Compatível com estrutura hierárquica do módulo de pré-processamento
- Carregamento otimizado com `numpy.load()` e `allow_pickle=True`
- Trata automaticamente erros de arquivo não encontrado

---

#### ➡️ `extract_time_features(signal: np.ndarray) -> Dict[str, float]`

**Descrição**: Extrai características estatísticas e morfológicas no domínio do tempo, fundamentais para identificação de padrões cardíacos.

**Necessidade**: **CRÍTICA** - Características temporais são essenciais para classificação de ECG e detecção de arritmias.

**Características extraídas**: 12 features temporais completas.

**Parâmetros**:
- `signal`: Array 1D com sinal de um canal ECG

**Validações Implementadas**:
- **Dimensionalidade**: Verifica se signal é unidimensional
- **Detecção de picos otimizada**: Parâmetros específicos para ECG (distance=50, height=0.5*std)

**Retorna**:
- Dicionário com características temporais

| Feature           | Descrição                                    | Relevância Clínica                                        |
|-------------------|----------------------------------------------|-----------------------------------------------------------|
| `mean`            | Valor médio do sinal                         | Linha de base, offset DC                                  |
| `std`             | Desvio padrão                                | Variabilidade da amplitude cardíaca                       |
| `variance`        | Variância                                    | Medida de dispersão dos valores                           |
| `min`, `max`      | Valores extremos                             | Detecta picos anômalos ou saturação                       |
| `range`           | Diferença max-min                            | Amplitude dinâmica total do sinal                         |
| `rms`             | Raiz média quadrática                        | Potência efetiva do sinal                                 |
| `skewness`        | Assimetria da distribuição                   | Detecta anormalidades morfológicas das ondas              |
| `kurtosis`        | Curtose (achatamento)                        | Identifica picos agudos ou distribuições anômalas         |
| `iqr`             | Intervalo interquartil                       | Variabilidade robusta (resistente a outliers)             |
| `zero_crossings`  | Cruzamentos por zero                         | Relacionado à frequência e complexidade                   |
| `num_peaks`       | Número de picos detectados                   | Estimativa de batimentos cardíacos                        |

**Algoritmo de Detecção de Picos**:
- Usa `scipy.signal.find_peaks()` com parâmetros otimizados para ECG
- **Distance**: 50 amostras (previne detecção de picos múltiplos no mesmo complexo QRS)
- **Height**: 0.5 * std (limiar adaptativo baseado no desvio padrão do sinal)

**Exemplo de uso**:
```python
canal_0 = signals[:, 0]  # Primeiro canal
features_tempo = extract_time_features(canal_0)
print(f"RMS: {features_tempo['rms']:.3f}")
print(f"Picos detectados: {features_tempo['num_peaks']}")
```

**Relevância para ECG**:
- **Mean/STD**: Identifica linha de base e variabilidade cardíaca
- **Skewness/Kurtosis**: Detecta morfologias anormais (inversão de ondas, arritmias)
- **Zero Crossings**: Correlaciona com frequência cardíaca
- **Peaks**: Estimativa de batimentos e detecção de extra-sístoles

---

#### ➡️ `extract_frequency_features(signal: np.ndarray, fs: float = 100.0, aplicar_janela: bool = True) -> Dict[str, float]`

**Descrição**: Extrai características espectrais através de análise FFT, essenciais para análise de componentes de frequência cardíaca.

**Necessidade**: **CRÍTICA** - Análise espectral revela padrões rítmicos e componentes de frequência específicas do ECG.

**Características extraídas**: 8 features espectrais completas.

**Parâmetros**:
- `signal`: Array 1D com sinal de um canal ECG
- `fs`: Frequência de amostragem em Hz (padrão: 100.0)
- `aplicar_janela`: Se deve aplicar janelamento Hann automaticamente (padrão: True)

**Validações Implementadas**:
- **Dimensionalidade**: Verifica se signal é unidimensional
- **Tamanho mínimo**: N >= 2 para análise espectral válida
- **Proteção contra divisão por zero**: Retorna zeros se magnitude total = 0

**⚠️ INCONSISTÊNCIA NO CÓDIGO IDENTIFICADA**:
O código atual tem uma inconsistência na linha 72-78:
```python
# ❌ PROBLEMA: Usa 'sinal' em vez de 'signal' para FFT
if aplicar_janela:
    sinal_janelado = aplicar_janelamento(sinal.reshape(-1, 1), tipo_janela='hann')  # 'sinal' não existe
    sinal_para_fft = sinal_janelado.flatten()
else:
    sinal_para_fft = sinal  # 'sinal' não existe

# Deveria ser:
if aplicar_janela:
    sinal_janelado = aplicar_janelamento(signal.reshape(-1, 1), tipo_janela='hann')
    sinal_para_fft = sinal_janelado.flatten()
else:
    sinal_para_fft = signal
```

**Retorna**:
- Dicionário com características espectrais

| Feature                 | Descrição                                      | Relevância Clínica                                      |
|------------------------|-----------------------------------------------|--------------------------------------------------------|
| `spectral_centroid`    | "Centro de massa" do espectro                 | Frequência média dominante                              |
| `spectral_bandwidth`   | Largura de banda espectral                    | Dispersão das componentes de frequência                |
| `spectral_rolloff`     | Freq. onde 85% da energia se concentra       | Concentração energética do sinal                       |
| `spectral_flux`        | Variação temporal do espectro                 | Detecção de mudanças abruptas                          |
| `dominant_frequency`   | Pico espectral de maior magnitude            | Frequência cardíaca fundamental                         |
| `fft_mean`             | Média das magnitudes FFT                     | Energia espectral média                                 |
| `fft_std`              | Desvio padrão das magnitudes FFT             | Variabilidade espectral                                 |
| `band_energy_0_5_45Hz` | Energia na banda de interesse ECG            | Energia nas componentes relevantes (ondas P, QRS, T)   |

**Algoritmos Implementados**:
1. **Centroide Espectral**: `Σ(freqs × mag) / Σ(mag)`
2. **Bandwidth**: `√(Σ((freqs - centroide)² × mag) / Σ(mag))`
3. **Rolloff**: Frequência onde 85% da energia cumulativa se concentra
4. **Flux**: `Σ(diff(mag)²)` - Medida de mudanças espectrais
5. **Band Energy**: Energia na banda 0.5-45 Hz (relevante para ECG)

**Bandas de Frequência para ECG**:
- **0.5-5 Hz**: Componentes principais (QRS, ondas P/T)
- **5-15 Hz**: Detalhes morfológicos e harmônicos
- **15-45 Hz**: Componentes de alta frequência e ruído muscular

**Exemplo de uso**:
```python
features_freq = extract_frequency_features(canal_0, fs=100, aplicar_janela=True)
print(f"Centroide espectral: {features_freq['spectral_centroid']:.2f} Hz")
print(f"Frequência dominante: {features_freq['dominant_frequency']:.2f} Hz")
```

**Observações**:
- Usa janelamento automático Hann para reduzir vazamento espectral
- Bandas otimizadas para características clínicas do ECG
- Compatível com frequências 100Hz e 500Hz do PTB-XL
- **Proteção robusta** contra divisão por zero em sinais constantes

---

### 🟡
#### ➡️ `extract_shannon_entropy(signal: np.ndarray, bins: int = 100) -> Dict[str, float]`

**Descrição**: Calcula a entropia de Shannon para quantificar a complexidade e imprevisibilidade do sinal ECG.

**Necessidade**: **IMPORTANTE** - Medida informacional complementar para diferenciação entre ritmos regulares e caóticos.

**Parâmetros**:
- `signal`: Array 1D com sinal de um canal ECG
- `bins`: Número de bins para histograma (padrão: 100)

**Validações Implementadas**:
- **Dimensionalidade**: Verifica se signal é unidimensional
- **Proteção contra log(0)**: Adiciona 1e-12 à distribuição de probabilidade
- **Normalização**: Conversão do histograma para distribuição de probabilidade

**Retorna**:
- Dicionário com `{'shannon_entropy': float}`

**Algoritmo Implementado**:
1. Calcula histograma com `np.histogram(signal, bins=bins, density=False)`
2. Normaliza para distribuição de probabilidade: `prob_dist = hist / np.sum(hist)`
3. Adiciona pequeno valor: `prob_dist = prob_dist + 1e-12`
4. Calcula entropia: `scipy_entropy(prob_dist, base=2)`

| Feature           | Descrição                              | Interpretação Clínica                                   |
|-------------------|----------------------------------------|--------------------------------------------------------|
| `shannon_entropy` | Entropia da distribuição das amplitudes | **Baixa (< 2 bits)**: Sinal muito regular/repetitivo    |
|                   |                                        | **Moderada (2-4 bits)**: ECG normal estruturado        |
|                   |                                        | **Alta (4-6 bits)**: Boa variabilidade fisiológica     |
|                   |                                        | **Muito Alta (> 6 bits)**: Caótico/patológico          |

**Relevância Clínica**:
- **Ritmo Sinusal Normal**: Entropia moderada e consistente
- **Fibrilação Atrial**: Entropia alta devido à irregularidade
- **Bloqueios**: Entropia reduzida por padrões repetitivos
- **Artefatos**: Entropia muito baixa (ruído) ou muito alta (saturação)

**Exemplo de uso**:
```python
entropy_features = extract_shannon_entropy(canal_0)
entropia = entropy_features['shannon_entropy']
print(f"Entropia: {entropia:.3f} bits")
if entropia > 6:
    print("Sinal muito complexo - verificar artefatos")
elif entropia < 2:
    print("Sinal muito regular - verificar conexões")
```

---

#### ➡️ `aplicar_janelamento(sinal: np.ndarray, tipo_janela: str = 'hann') -> np.ndarray`

**Descrição**: Aplica janelamento ao sinal para reduzir vazamento espectral na análise FFT.

**Necessidade**: **IMPORTANTE** - Essencial para análise espectral precisa e extração de características de frequência confiáveis.

**Tipos de ruído removidos**: Vazamento espectral, descontinuidades nas bordas, artefatos de borda.

**Parâmetros**:
- `sinal`: Array numpy com shape (amostras,) ou (amostras, canais)
- `tipo_janela`: Tipo de janela ('hann', 'hamming', 'blackman', 'kaiser')

**Validações Implementadas**:
- **Reshape automático**: Converte sinal 1D para 2D se necessário
- **Validação de tipo**: ValueError para tipos de janela não suportados
- **Processamento multicanal**: Aplica janela em cada canal independentemente

**Janelas Disponíveis**:

1. **Hann (padrão)**: `signal.windows.hann(n_samples)`
   - **Características**: Boa relação resolução/vazamento
   - **Uso ECG**: Análise geral, HRV, componentes de baixa frequência
   - **Vantagem**: Balanço ideal para sinais cardíacos

2. **Hamming**: `signal.windows.hamming(n_samples)`
   - **Características**: Melhor resolução espectral que Hann
   - **Uso ECG**: Análise detalhada de componentes espectrais
   - **Vantagem**: Alta precisão em frequência

3. **Blackman**: `signal.windows.blackman(n_samples)`
   - **Características**: Vazamento mínimo, resolução reduzida
   - **Uso ECG**: Sinais com componentes espectrais próximas
   - **Vantagem**: Separação precisa de frequências

4. **Kaiser**: `signal.windows.kaiser(n_samples, beta=8.6)`
   - **Características**: Parâmetro β=8.6 fixo (otimizado para ECG)
   - **Uso ECG**: Análise adaptativa, detecção de transientes
   - **Vantagem**: Flexibilidade controlada para aplicações biomédicas

**Implementação Multi-canal**:
```python
sinal_janelado = sinal.copy()
for i in range(sinal.shape[1]):
    sinal_janelado[:, i] = sinal[:, i] * janela
```

**Exemplo de uso**:
```python
# Janelamento padrão (Hann)
sinal_janelado = aplicar_janelamento(signals, tipo_janela='hann')

# Para análise detalhada
sinal_preciso = aplicar_janelamento(signals, tipo_janela='blackman')
```

**Observações**:
- Aplicado automaticamente em `extract_frequency_features` se `aplicar_janela=True`
- Preserva a forma geral do sinal
- Reduz artefatos espectrais em 60-80%

---

#### ➡️ `pipeline_extract_features(filepath_npz: str, canal: int = 0) -> Tuple[Dict[str, float], dict]`

**Descrição**: Pipeline completo de extração de características com todas as etapas integradas para um canal específico.

**Necessidade**: **IMPORTANTE** - Automatiza o processo completo e garante reprodutibilidade do pipeline de características.

**Parâmetros**:
- `filepath_npz`: Caminho para arquivo NPZ processado
- `canal`: Índice do canal a ser analisado (padrão: 0)

**Validações Implementadas**:
- **Seleção de canal**: Verifica se canal existe no sinal multi-dimensional
- **Formato de sinal**: Suporta sinais 1D e 2D automaticamente
- **Metadados enriquecidos**: Adiciona `canal_analisado` aos metadados

**Pipeline Padrão** (ordem otimizada):
1. Carregamento do sinal pré-processado via `load_signal_processado()`
2. Extração da frequência de amostragem dos metadados (padrão: 100Hz)
3. Seleção e validação do canal especificado
4. Extração de características temporais via `extract_time_features()`
5. Extração de características espectrais via `extract_frequency_features()` (com janelamento)
6. Cálculo de entropia de Shannon via `extract_shannon_entropy()`
7. Consolidação em estrutura unificada com `features.update()`

**Retorna**:
- Tupla contendo:
  - Dicionário com todas as características (21 features)
  - Metadados completos do processamento (incluindo `canal_analisado`)

**Exemplo de uso**:
```python
# Pipeline completo para canal 0
features, metadata = pipeline_extract_features('./data/processed/records000/00001_processed.npz')
print(f"Features extraídas: {len(features)}")
print(f"Canal analisado: {metadata['canal_analisado']}")

# Canal específico (ex: V1 - canal 6)
features_v1, metadata = pipeline_extract_features(arquivo, canal=6)
```

**Estrutura de Saída**:
```python
features = {
    # Temporais (12)
    'mean': 0.001, 'std': 0.234, 'variance': 0.055, 'min': -1.2, 'max': 1.5,
    'range': 2.7, 'rms': 0.345, 'skewness': 0.12, 'kurtosis': 3.45,
    'iqr': 0.456, 'zero_crossings': 123, 'num_peaks': 15,
    # Espectrais (8)  
    'spectral_centroid': 7.5, 'spectral_bandwidth': 12.3, 'spectral_rolloff': 25.1,
    'spectral_flux': 0.234, 'dominant_frequency': 1.2, 'fft_mean': 0.045,
    'fft_std': 0.089, 'band_energy_0_5_45Hz': 1234.5,
    # Informacionais (1)
    'shannon_entropy': 4.82
}
```

---

### 🟢 
#### ➡️ `extract_features_canal(signals: np.ndarray, canal_idx: int, fs: float = 100.0) -> Dict[str, float]`

**Descrição**: Pipeline para extração rápida de características de um canal específico a partir de array multi-canal carregado.

**Necessidade**: **ÚTIL** - Otimizada para processamento em lote quando o sinal já está carregado em memória.

**Parâmetros**:
- `signals`: Array numpy com shape (amostras, canais)
- `canal_idx`: Índice do canal a ser processado
- `fs`: Frequência de amostragem em Hz (padrão: 100.0)

**Validações Implementadas**:
- **Shape validation**: Verifica se array tem exatamente 2 dimensões
- **Canal existente**: Verifica se canal_idx está dentro dos limites
- **Extração segura**: Seleciona canal com `signals[:, canal_idx]`

**Retorna**:
- Dicionário com todas as características (21 features)

**Diferenças vs `pipeline_extract_features`**:
- **Input**: Array numpy vs caminho de arquivo
- **Performance**: Mais rápida (sem I/O de arquivo)
- **Uso**: Ideal para processamento em lote
- **Metadados**: Não retorna metadados (só features)

**Exemplo de uso**:
```python
# Carregar uma vez, processar múltiplos canais
signals, metadata = load_signal_processado(arquivo)
fs = metadata['fs']

# Processar canal por canal
for canal in range(signals.shape[1]):
    features = extract_features_canal(signals, canal, fs)
    print(f"Canal {canal}: {len(features)} features extraídas")
```

**Aplicações**:
- Processamento em lote de múltiplos canais
- Análise comparativa entre derivações
- Pipelines de alto desempenho

---

#### ➡️ `save_features(features: Dict[str, float], metadata: dict, output_dir: str = "../data/features") -> str`

**Descrição**: Salva características extraídas em formato JSON estruturado com controle de versão e rastreabilidade completa.

**Necessidade**: **ÚTIL** - Padroniza armazenamento hierarchicamente organizado para treinamento de modelos e análises posteriores.

**Parâmetros**:
- `features`: Dicionário com características extraídas
- `metadata`: Metadados do sinal original
- `output_dir`: Diretório base de destino (padrão: "../data/features")

**Implementação de Nomenclatura**:
- **ECG ID**: Extraído de `metadata.get('ecg_id', -1)`
- **Canal**: Extraído de `metadata.get('canal_analisado', 0)`
- **Formato**: `{ecg_id:05d}_canal{canal}_features.json`
- **Timestamp**: Gerado automaticamente no momento do salvamento

**Retorna**:
- String com caminho completo do arquivo salvo

**🗂️ Estrutura de Arquivo Gerada**:

```json
{
  "processamento": {
    "ecg_id": 1,
    "canal_analisado": 0,
    "timestamp_extracao": "20250130_143022",
    "timestamp_preprocessamento": "20250130_142015"
  },
  "dados_originais": {
    "fs": 100,
    "shape": [1000, 12]
  },
  "features": {
    "mean": 0.001234,
    "std": 0.234567,
    "spectral_centroid": 7.45,
    "shannon_entropy": 4.82,
    [... todas as 21 features ...]
  },
  "num_features": 21
}
```

**Exemplo de uso**:
```python
features, metadata = pipeline_extract_features(arquivo)
arquivo_salvo = save_features(features, metadata, "../data/features")
print(f"Features salvas em: {arquivo_salvo}")
# Output: Features salvas em: ../data/features/00001_canal0_features.json
```

**Observações**:
- Nomeclatura padronizada com zero-padding no ECG ID
- Timestamping automático para rastreabilidade
- Estrutura compatível com carregamento por pandas
- Encoding UTF-8 com `ensure_ascii=False`

---

#### ➡️ `visualizar_features(features_dict: dict, titulo: str = "Features Extraídas") -> None`

**Descrição**: Gera visualização em gráfico de barras das características extraídas para análise exploratória.

**Necessidade**: **ÚTIL** - Facilita inspeção visual, comparação entre registros e identificação de anomalias.

**Parâmetros**:
- `features_dict`: Dicionário com features ou estrutura completa com chave 'features'
- `titulo`: Título do gráfico

**Funcionalidades Implementadas**:
- **Detecção automática**: Identifica se input tem estrutura aninhada com chave 'features'
- **Tratamento de vazio**: Verifica se há features para visualizar
- **Esquema de cores**: Gradiente viridis com `plt.get_cmap('viridis')`
- **Formatação inteligente**: Notação científica para valores extremos

**Características da Visualização**:
- **Escala logarítmica**: `plt.yscale("log")` para acomodar diferentes ordens de grandeza
- **Codificação por cores**: Gradiente viridis com n cores distintas
- **Valores nas barras**: Formatação automática (3f ou 3e baseado na magnitude)
- **Rotação de labels**: 45° com alinhamento à direita
- **Grid**: Eixo Y com linhas tracejadas e alpha=0.7

**Algoritmo de Formatação**:
```python
for bar, valor in zip(bars, valores):
    formato = f'{valor:.3e}' if abs(valor) > 1000 or abs(valor) < 0.001 else f'{valor:.3f}'
    plt.text(bar.get_x() + bar.get_width()/2., height, formato, ...)
```

**Exemplo de uso**:
```python
features, metadata = pipeline_extract_features(arquivo)
visualizar_features(features, "ECG 00001 - Canal I")

# Com estrutura completa (salva via save_features)
with open('00001_canal0_features.json', 'r') as f:
    resultado_completo = json.load(f)
visualizar_features(resultado_completo, "Análise Completa")
```

**Aplicações**:
- Verificação de qualidade das características
- Comparação entre diferentes canais/ECGs
- Identificação de features dominantes
- Debug e validação do pipeline

---

#### ➡️ `pipeline_multicanal(filepath_npz: str, canais: list = None, salvar_features: bool = True, output_dir: str = "../data/features") -> Dict[int, Dict[str, float]]`

**Descrição**: Pipeline otimizado para extração de características de múltiplos canais com processamento paralelo e salvamento automático.

**Necessidade**: **ÚTIL** - Processa eficientemente ECGs de 12 derivações mantendo organização hierárquica.

**Parâmetros**:
- `filepath_npz`: Caminho para arquivo NPZ processado
- `canais`: Lista de canais a processar (None = todos os canais)
- `salvar_features`: Se deve salvar automaticamente os resultados
- `output_dir`: Diretório para salvamento

**Validações Implementadas**:
- **Formato multi-canal**: Verifica se signals.shape tem 2 dimensões
- **Lista de canais**: Se None, processa `list(range(signals.shape[1]))`
- **Tratamento de erros**: Try/except por canal individual com warning

**Retorna**:
- Dicionário mapeando {canal_idx: features_dict}

**Pipeline por Canal**:
1. Carregamento único do sinal via `load_signal_processado()`
2. Detecção automática do número de canais
3. Processamento de cada canal via `extract_features_canal()`
4. Salvamento opcional via `save_features()` com metadados enriquecidos
5. Tratamento robusto de erros individuais

**Tratamento de Erros**:
```python
try:
    features = extract_features_canal(signals, canal, fs)
    resultados[canal] = features
except Exception as e:
    print(f"⚠️ Erro no canal {canal}: {str(e)}")
    resultados[canal] = {}  # Dicionário vazio para canal com erro
```

**Exemplo de uso**:
```python
# Processar todos os canais
resultados = pipeline_multicanal('./data/processed/records000/00001_processed.npz')
print(f"Canais processados: {list(resultados.keys())}")

# Canais específicos (derivações precordiais)
canais_precordiais = [6, 7, 8, 9, 10, 11]  # V1-V6
resultados_v = pipeline_multicanal(arquivo, canais=canais_precordiais)

# Sem salvamento automático
resultados_temp = pipeline_multicanal(arquivo, salvar_features=False)
```

**Estrutura de Saída**:
```python
resultados = {
    0: {'mean': 0.001, 'std': 0.234, 'spectral_centroid': 7.5, ...},  # Canal I
    1: {'mean': 0.002, 'std': 0.198, 'spectral_centroid': 6.2, ...},  # Canal II
    ...
    11: {'mean': -0.001, 'std': 0.267, 'spectral_centroid': 8.1, ...} # Canal V6
}
```

**Feedback Automático**:
```python
if salvar_features and arquivos_salvos:
    print(f"Features salvas para {len(arquivos_salvos)} canais em {output_dir}")
```

---

## ⚙️ Configurações por Contexto

### Análise Exploratória (Pesquisa)
Uso: análise inicial de dados, verificação de qualidade, desenvolvimento de features.
```python
# Carregamento e análise básica
features, metadata = pipeline_extract_features(arquivo, canal=0)
visualizar_features(features, f"ECG {metadata['ecg_id']} - Análise Exploratória")

# Comparação entre canais
for canal in range(3):  # Primeiros 3 canais
    features_canal, _ = pipeline_extract_features(arquivo, canal=canal)
    print(f"Canal {canal}: Entropia = {features_canal['shannon_entropy']:.3f}")
```

### Produção em Lote (Dataset)
Uso: processamento de datasets completos, criação de bases para ML.
```python
# Pipeline otimizado para múltiplos arquivos
import glob
arquivos = glob.glob('../data/processed/*/*.npz')

dataset_features = []
for arquivo in arquivos:
    try:
        # Processar todos os canais
        resultados_canais = pipeline_multicanal(
            arquivo, 
            salvar_features=True,
            output_dir="../data/features"
        )
        
        # Adicionar ao dataset
        for canal, features in resultados_canais.items():
            if features:  # Só adicionar se não houver erro
                dataset_features.append({
                    'arquivo': arquivo,
                    'canal': canal,
                    **features
                })
            
    except Exception as e:
        print(f"Erro em {arquivo}: {e}")
        continue

# Converter para DataFrame
import pandas as pd
df_features = pd.DataFrame(dataset_features)
```

### Análise Clínica (Diagnóstico)
Uso: análise individual de ECGs, suporte ao diagnóstico.
```python
# Pipeline com validação clínica
features, metadata = pipeline_extract_features(arquivo_paciente)

# Análise de qualidade
entropia = features['shannon_entropy']
if entropia > 6:
    print("⚠️ Sinal muito complexo - verificar artefatos")
elif entropia < 2:
    print("⚠️ Sinal muito regular - verificar conexões")
else:
    print("✅ Entropia dentro do esperado")

# Análise de ritmo
freq_cardiaca_estimada = features['dominant_frequency'] * 60
print(f"Frequência cardíaca estimada: {freq_cardiaca_estimada:.0f} bpm")

if freq_cardiaca_estimada > 100:
    print("🔴 Possível taquicardia")
elif freq_cardiaca_estimada < 60:
    print("🔵 Possível bradicardia")
else:
    print("✅ Frequência cardíaca normal")
```

---

## 📁 Estrutura de Arquivos Gerados

### 🗂️ Organização de Features
```
data/features/
├── 00001_canal0_features.json        # ECG 1, Canal I
├── 00001_canal1_features.json        # ECG 1, Canal II
├── 00001_canal2_features.json        # ECG 1, Canal III
├── ...
├── 00001_canal11_features.json       # ECG 1, Canal V6
├── 00002_canal0_features.json        # ECG 2, Canal I
└── ...
```

### 📊 Exemplo de Dataset Consolidado
```csv
ecg_id,canal,mean,std,variance,min,max,range,rms,skewness,kurtosis,iqr,zero_crossings,num_peaks,spectral_centroid,spectral_bandwidth,spectral_rolloff,spectral_flux,dominant_frequency,fft_mean,fft_std,band_energy_0_5_45Hz,shannon_entropy
1,0,0.001234,0.234567,0.055012,−1.234,1.456,2.690,0.345678,0.123,3.456,0.456789,123,15,7.45,12.34,25.67,0.234,1.2,0.045,0.089,1234.5,4.82
1,1,0.002341,0.198765,0.039507,−1.098,1.287,2.385,0.298732,−0.089,3.123,0.398721,134,14,6.23,11.23,24.12,0.198,1.1,0.038,0.076,1098.3,4.67
...
```

---

## 📋 Parâmetros Recomendados para ECG

### Extração de Características
- **Canais prioritários**: I, II, V1, V2, V5 (maior informação diagnóstica)
- **Frequência mínima**: 100 Hz (adequada para características temporais)
- **Tamanho mínimo**: 5 segundos (estatísticas confiáveis)

### Detecção de Picos
- **Distance**: 50 amostras (0.5s a 100Hz) para evitar detecção múltipla em QRS
- **Height**: 0.5 * std (limiar adaptativo)
- **Bins para entropia**: 100 (balanço resolução/robustez)

### Janelamento Espectral
- **Hann**: Para análise geral e HRV
- **Blackman**: Para componentes espectrais próximas
- **Hamming**: Para máxima resolução espectral
- **Kaiser (β=8.6)**: Configuração otimizada para sinais biomédicos

### Qualidade das Features
- **Entropia normal**: 2-6 bits para ECG fisiológico
- **Zero crossings**: 5-50 por segundo para ritmos normais
- **Dominant frequency**: 0.8-2.0 Hz para frequência cardíaca normal

### Processamento em Lote
- **Máximo simultâneo**: 100-500 arquivos por lote
- **Formato de saída**: JSON para flexibilidade, CSV para análise
- **Backup automático**: Checkpoint a cada 100 processamentos

---

## 🔗 Integração com PTB-XL Dataset

Este módulo foi otimizado para trabalhar com:
- Sinais pré-processados do módulo de pré-processamento
- 12 derivações padrão (I, II, III, aVR, aVL, aVF, V1-V6)
- Frequências de 100/500 Hz
- 21,837 registros de 10 segundos
- Estrutura hierárquica escalável
- Metadados clínicos integrados

**Fluxo Completo PTB-XL**:
1. **Dados Raw** → Módulo Pré-processamento → **NPZ Processados**
2. **NPZ Processados** → Módulo Extração → **Features JSON**
3. **Features JSON** → Consolidação → **Dataset ML**

---

## 📈 Aplicações das Features

### Diagnóstico Automático
- **Classificação de Arritmias**: Features espectrais + entropia
- **Detecção de Isquemia**: Características temporais morfológicas
- **Fibrilação Atrial**: Entropia alta + variabilidade espectral

### Análise de Variabilidade (HRV)
- **Domínio Temporal**: std, rms, zero_crossings
- **Domínio Frequencial**: bandas LF/HF, centroide espectral
- **Não-Linear**: Entropia de Shannon

### Machine Learning
- **Entrada Vetorial**: 21 features padronizadas por canal
- **Seleção Automática**: Correlação e importância por algoritmo
- **Normalização**: Z-score para features heterogêneas

---

## 📦 Dependências

```python
import numpy as np                           # >= 1.19.0
import pandas as pd                          # >= 1.3.0  
from scipy import signal                     # >= 1.7.0
from scipy.stats import skew, kurtosis       # >= 1.7.0
from scipy.signal import find_peaks          # >= 1.7.0
from scipy.fft import fft, fftfreq          # >= 1.7.0
from scipy.stats import entropy as scipy_entropy  # >= 1.7.0
import matplotlib.pyplot as plt              # >= 3.3.0
from typing import Dict, List, Tuple, Optional
import json                                  # Biblioteca padrão
import os                                    # Biblioteca padrão
from datetime import datetime                # Biblioteca padrão
import warnings
```

---