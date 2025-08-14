# Módulo de Classificação de Sinais ECG

Este módulo contém funções essenciais para classificação automatizada de sinais ECG, especificamente otimizado para trabalhar com características extraídas e diagnósticos do dataset PTB-XL.

## 📊 Classificação por Necessidade

**🔴 CRÍTICAS**: `ECGRandomForestClassifier`, `load_features_dataset`, `load_labels_ptbxl`  
**🟡 IMPORTANTES**: `preparar_dados_por_paciente`, `determinar_classe_principal`, `find_ptbxl_database`  
**🟢 ÚTEIS**: `plot_classification_results`, `save_model_and_results`, `plot_feature_importance`, `find_latest_features_file`

---

## 🎯 Classes Alvo do Projeto

### Mapeamento Clínico Implementado
```python
CLASSES_ALVO = {
    'MI': 'Infarto do Miocárdio',
    'AFIB': 'Fibrilação Atrial',
    'NORM': 'Normal/Saudável'
}

MAPEAMENTO_PTB_XL = {
    'MI': ['IMI', 'ASMI', 'LMI', 'PMI', 'AMI'],   # Tipos de infarto
    'AFIB': ['AFIB', 'AFLT'],                     # Fibrilação atrial
    'NORM': ['NORM', 'SR']                        # Normal e ritmo sinusal
}
```

**Descrição das Classes quanto ao Sinal ECG**:
- **MI (Infarto)**: Alterações no segmento ST (elevação ou depressão), inversão de ondas T, presença de ondas Q patológicas;
- **AFIB (Fibrilação)**: Ausência de ondas P distintas, intervalos RR irregulares, atividade elétrica atrial desorganizada;
- **NORM (Normal)**: Presença de ondas P regulares antes de cada complexo QRS, intervalos PR e QRS dentro dos valores de referência, ritmo sinusal estável;

---

## Funções Principais

### 🔴 CRÍTICAS
#### ➡️ `load_features_dataset(filepath: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]`

**Descrição**: Carrega dataset de características extraídas com detecção automática de estrutura.

**Necessidade**: **CRÍTICA** - Ponte entre extração de features (Notebook II) e treinamento de modelos.

**Parâmetros**:
- `filepath`: Caminho para arquivo CSV (None = busca automática)

**Retorna**:
- Tupla com DataFrame das características e dicionário de metadados

**Funcionalidades Implementadas**:
- **Busca automática**: Usa `find_latest_features_file()` se filepath=None
- **Detecção de estrutura**: Identifica automaticamente colunas de canal
- **Metadados completos**: Shape, ECGs únicos, colunas disponíveis
- **Validação de integridade**: Verifica se arquivo existe e é válido

**Metadados Gerados**:
```python
metadata = {
    'filepath': '/path/to/features.csv',
    'total_records': 12000,
    'unique_ecgs': 1000, 
    'columns': ['ecg_id', 'canal', 'mean', 'std', ...],
    'channel_column': 'canal',
    'channels_per_ecg': 12
}
```

**Exemplo de uso**:
```python
df_features, metadata = load_features_dataset()
print(f"Features carregadas: {df_features.shape}")
print(f"ECGs únicos: {metadata['unique_ecgs']}")
```

---

#### ➡️ `load_labels_ptbxl(ptbxl_path: str = None, ecg_ids: List[int] = None) -> pd.DataFrame`

**Descrição**: Carrega e mapeia diagnósticos clínicos do PTB-XL para nossas 3 classes.

**Necessidade**: **CRÍTICA** - Converte códigos SCP médicos para labels de machine learning.

**Parâmetros**:
- `ptbxl_path`: Caminho para ptbxl_database.csv (None = busca automática)
- `ecg_ids`: Lista de ECGs para filtrar (None = todos)

**Retorna**:
- DataFrame com colunas `['ecg_id', 'classe']`

**Mapeamento Clínico Implementado**:
- **Busca hierárquica**: Usa `find_ptbxl_database()` se ptbxl_path=None
- **Filtragem inteligente**: Processa apenas ECGs disponíveis em ecg_ids
- **Priorização médica**: MI > AFIB > NORM (ordem de importância clínica)
- **Estatísticas automáticas**: Mostra distribuição de classes

**Processamento de Diagnósticos**:
1. Carrega CSV PTB-XL completo
2. Filtra apenas ECGs com features disponíveis
3. Aplica `determinar_classe_principal()` em cada scp_codes
4. Remove casos sem diagnóstico claro
5. Retorna DataFrame limpo

**Exemplo de uso**:
```python
ecg_ids = [1, 2, 3, 1000, 2500]  # ECGs processados
df_labels = load_labels_ptbxl(ecg_ids=ecg_ids)
print(df_labels['classe'].value_counts())
```

---

#### ➡️ `ECGRandomForestClassifier`

**Descrição**: Classe principal para classificação ECG usando Random Forest otimizado.

**Necessidade**: **CRÍTICA** - Encapsula todo pipeline de treinamento, avaliação e predição.

**Atributos da Classe**:
- `config`: Configuração do Random Forest
- `model`: Modelo treinado (RandomForestClassifier)
- `scaler`: StandardScaler para normalização
- `label_encoder`: LabelEncoder para classes
- `training_history`: Histórico do treinamento

**Configuração Padrão**:
```python
RF_CONFIG_DEFAULT = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',  # Lida com desbalanceamento
    'random_state': 42,
    'n_jobs': -1  # Paralelização completa
}
```

**Métodos Principais**:

1. **`prepare_data(df_features, df_labels) -> Tuple[np.ndarray, np.ndarray]`**:
   - Faz merge de features e labels por ecg_id
   - Extrai matriz X de features e vetor y de classes
   - Aplica StandardScaler e LabelEncoder
   - Retorna dados prontos para treinamento

2. **`split_data(X, y, test_size=0.2, random_state=42) -> Tuple`**:
   - Divisão estratificada treino/teste
   - Mantém proporção de classes em ambos conjuntos

3. **`train(X_train, y_train, optimize_hyperparams=True, cv_folds=5) -> Dict`**:
   - Treina com Grid Search opcional
   - Validação cruzada estratificada
   - Salva histórico completo do treinamento

4. **`predict(X) -> np.ndarray`**:
   - Predições simples (classe mais provável)

5. **`predict_proba(X) -> Dict[str, np.ndarray]`**:
   - Retorna probabilidades para todas as classes
   - Inclui mapeamento de classes

6. **`evaluate(X_test, y_test) -> Dict[str, Any]`**:
   - Avaliação completa com múltiplas métricas
   - Matriz de confusão, relatório detalhado
   - Predições e probabilidades salvas

**Grid de Otimização**:
```python
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
```

**Exemplo de uso completo**:
```python
# Inicializar classificador
classifier = ECGRandomForestClassifier()

# Preparar dados
X, y = classifier.prepare_data(df_features, df_labels)

# Dividir dados
X_train, X_test, y_train, y_test = classifier.split_data(X, y)

# Treinar com otimização
results = classifier.train(X_train, y_train, optimize_hyperparams=True)

# Avaliar
metrics = classifier.evaluate(X_test, y_test)
```

---

### 🟡 IMPORTANTES
#### ➡️ `preparar_dados_por_paciente(df_features: pd.DataFrame) -> pd.DataFrame`

**Descrição**: Converte dados por canal (12 derivações) para dados por paciente (vetor único).

**Necessidade**: **IMPORTANTE** - Machine learning requer um vetor por paciente, não por canal.

**Transformação Implementada**:
- **Detecção automática**: Identifica se dados já estão por paciente
- **Concatenação inteligente**: Une features de 12 canais em vetor único
- **Limpeza automática**: Remove colunas auxiliares (canal, fs, n_samples, etc.)
- **Validação de estrutura**: Verifica integridade dos dados

**Algoritmo de Detecção**:
1. Verifica se existe coluna 'ecg_id'
2. Detecta coluna de canal ('canal', 'channel', 'lead', 'derivacao')
3. Identifica colunas auxiliares para remover
4. Compara unique_ecgs vs total_records para detectar estrutura

**Antes vs Depois**:
```python
# ANTES: 12 registros por paciente (um por canal)
# ecg_id | canal | feature_1 | feature_2 | ...
#   1001 |   0   |   0.123   |   4.567   | ...
#   1001 |   1   |   0.234   |   5.678   | ...
#   ...  |  ...  |    ...    |    ...    | ...

# DEPOIS: 1 registro por paciente (todas as features concatenadas)
# ecg_id | features (array com 12 * n_features elementos)
#   1001 | [0.123, 4.567, ..., 0.234, 5.678, ..., ...]
```

**Processamento Detalhado**:
```python
for ecg_id in df_limpo['ecg_id'].unique():
    dados_ecg = df_limpo[df_limpo['ecg_id'] == ecg_id]
    features_canais = dados_ecg.drop(columns=['ecg_id']).values
    features_concatenadas = features_canais.flatten()  # Vetor único
    dados_pacientes.append([ecg_id, features_concatenadas])
```

---

#### ➡️ `determinar_classe_principal(scp_codes: str, mapeamento: Dict[str, List[str]]) -> str`

**Descrição**: Mapeia códigos SCP do PTB-XL para nossas classes principais.

**Necessidade**: **IMPORTANTE** - Converte terminologia médica para labels ML.

**Parâmetros**:
- `scp_codes`: String ou dict com códigos SCP (ex: "{'IMI': 100, 'NORM': 50}")
- `mapeamento`: Dicionário MAPEAMENTO_PTB_XL

**Retorna**:
- String com classe ('MI', 'AFIB', 'NORM') ou None

**Lógica Clínica Implementada**:
1. **Prioridade médica**: MI > AFIB > NORM (ordem de importância)
2. **Robustez**: Trata casos vazios, None e malformados
3. **Flexibilidade**: Aceita string ou dict como entrada
4. **Segurança**: Try/catch para códigos corrompidos

**Algoritmo**:
```python
# 1. Validação de entrada
if pd.isna(scp_codes) or scp_codes == '{}':
    return None

# 2. Conversão para dict
if isinstance(scp_codes, str):
    codes_dict = eval(scp_codes)  # Conversão segura

# 3. Busca por prioridade
for classe, codigos in mapeamento.items():  # MI primeiro, depois AFIB, depois NORM
    for codigo in codigos:
        if codigo in codes_dict:
            return classe
```

**Exemplo de códigos**:
```python
# Entrada PTB-XL
scp_codes = "{'IMI': 100, 'NORM': 50}"  # Inferior MI + Normal

# Processamento
classe = determinar_classe_principal(scp_codes, MAPEAMENTO_PTB_XL)
# Resultado: 'MI' (prioridade médica)
```

---

#### ➡️ `find_ptbxl_database(data_path: str = "../data/") -> str`

**Descrição**: Localiza automaticamente o arquivo ptbxl_database.csv em múltiplas localizações.

**Necessidade**: **IMPORTANTE** - Dataset PTB-XL pode estar em diferentes estruturas de pastas.

**Busca Hierárquica**:
```python
possible_paths = [
    os.path.join(data_path, "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv"),
    os.path.join(data_path, "raw", "ptbxl_database.csv"),
    os.path.join(data_path, "ptbxl_database.csv"),
]
```

**Retorna**:
- String com caminho completo do arquivo encontrado
- Raises FileNotFoundError se não encontrar

---

### 🟢 ÚTEIS
#### ➡️ `find_latest_features_file(data_path: str = "../data/") -> str`

**Descrição**: Busca automaticamente o arquivo de features mais recente no diretório.

**Necessidade**: **ÚTIL** - Facilita carregamento automático sem especificar arquivo.

**Algoritmo de Busca**:
1. Verifica se diretório `data/features/` existe
2. Lista arquivos CSV que contenham 'features' no nome
3. Ordena por nome (assumindo timestamp no nome)
4. Retorna o mais recente

**Exemplo de uso**:
```python
arquivo = find_latest_features_file()
# Resultado: "../data/features/features_extracted_20250130_143022.csv"
```

---

#### ➡️ `plot_classification_results(metrics: Dict[str, Any], save_path: Optional[str] = None) -> None`

**Descrição**: Gera visualizações completas dos resultados de classificação.

**Necessidade**: **ÚTIL** - Facilita interpretação clínica e análise de performance.

**Visualizações Geradas** (figura 2x2):
1. **Matriz de Confusão**: Heatmap com valores absolutos
2. **Métricas por Classe**: Barras agrupadas (precision, recall, f1-score)
3. **Distribuição de Probabilidades**: Histogramas por classe
4. **Métricas Gerais**: Barras com valores nas barras

**Características**:
- **Layout profissional**: 2x2 subplots com título geral
- **Cores intuitivas**: Blues para confusão, cores distintas para métricas
- **Valores nas barras**: Formatação automática com 3 casas decimais
- **Salvamento opcional**: Alta resolução (300 DPI)

**Exemplo de uso**:
```python
# Após avaliação do modelo
metrics = classifier.evaluate(X_test, y_test)

# Gerar visualizações
plot_classification_results(metrics, save_path="results.png")
```

---

#### ➡️ `plot_feature_importance(importance_dict: Dict[str, float], save_path: Optional[str] = None) -> None`

**Descrição**: Visualiza importância das características para interpretabilidade.

**Necessidade**: **ÚTIL** - Identifica quais features são mais relevantes clinicamente.

**Funcionalidades**:
- **Top 20**: Mostra apenas características mais importantes
- **Ordenação automática**: Por ordem decrescente de importância
- **Layout horizontal**: Barras horizontais para melhor legibilidade
- **Valores numéricos**: Adiciona valores das importâncias nas barras
- **Salvamento opcional**: Para relatórios

**Exemplo de uso**:
```python
# Extrair importâncias do modelo treinado
if hasattr(classifier.model, 'feature_importances_'):
    importance_dict = {f'feature_{i}': imp 
                      for i, imp in enumerate(classifier.model.feature_importances_)}
    plot_feature_importance(importance_dict)
```

---

#### ➡️ `save_model_and_results(classifier: ECGRandomForestClassifier, metrics: Dict[str, Any], output_dir: str = "../results/classification") -> Dict[str, str]`

**Descrição**: Salva modelo treinado e resultados com rastreabilidade completa.

**Necessidade**: **ÚTIL** - Persistência para uso posterior e auditoria.

**Arquivos Gerados** (timestamp automático):
1. **`ecg_random_forest_model_{timestamp}.pkl`**: Modelo completo
2. **`evaluation_metrics_{timestamp}.json`**: Métricas estruturadas
3. **`classification_report_{timestamp}.txt`**: Relatório legível

**Estrutura do Modelo Salvo**:
```python
model_data = {
    'classifier': classifier,  # Objeto completo
    'training_history': classifier.training_history,
    'metadata': {
        'timestamp': '20250130_143022',
        'classes': ['MI', 'AFIB', 'NORM'],
        'model_type': 'RandomForest',
        'n_features': 252
    }
}
```

**Tratamento de Arrays NumPy**:
- Converte arrays para listas no JSON
- Mantém estrutura do classification_report
- Usa `default=str` para objetos não serializáveis

**Retorna**:
```python
{
    'model_path': '../results/classification/ecg_random_forest_model_20250130_143022.pkl',
    'metrics_path': '../results/classification/evaluation_metrics_20250130_143022.json',
    'report_path': '../results/classification/classification_report_20250130_143022.txt'
}
```

---


## 📊 Pipeline Completo de Classificação

### Fluxo Integrado
```python
# 1. Carregamento automático
df_features, metadata = load_features_dataset()  # Features do Notebook II
df_labels = load_labels_ptbxl(ecg_ids=df_features['ecg_id'].unique())

# 2. Preparação por paciente
df_pacientes = preparar_dados_por_paciente(df_features)

# 3. Classificação
classifier = ECGRandomForestClassifier()
X, y = classifier.prepare_data(df_pacientes, df_labels)
X_train, X_test, y_train, y_test = classifier.split_data(X, y)

# 4. Treinamento otimizado
results = classifier.train(X_train, y_train, optimize_hyperparams=True)

# 5. Avaliação completa
metrics = classifier.evaluate(X_test, y_test)

# 6. Visualização e salvamento
plot_classification_results(metrics)
save_model_and_results(classifier, metrics)
```

---

## 📋 Métricas de Avaliação Implementadas

### Métricas Básicas
- **Accuracy**: Acurácia geral
- **Balanced Accuracy**: Acurácia balanceada (importante para classes desbalanceadas)
- **F1-Score Weighted**: F1 ponderado por suporte de classe
- **Cohen's Kappa**: Concordância além do acaso

### Métricas por Classe
- **Precision**: Verdadeiros positivos / (VP + Falsos positivos)
- **Recall (Sensibilidade)**: VP / (VP + Falsos negativos)
- **F1-Score**: Média harmônica de precision e recall
- **Support**: Número de amostras por classe

### Análise de Confusão
- **Matriz de Confusão**: Padrões de erro entre classes
- **Probabilidades**: Distribuição de confiança das predições

---

## 🏥 Considerações Clínicas

### Importância por Classe
1. **MI (Infarto)**: 
   - **Métrica crítica**: Recall (não perder casos)
   - **Falso negativo**: Risco de vida
   - **Falso positivo**: Ansiedade, mas seguro

2. **AFIB (Fibrilação)**: 
   - **Métrica crítica**: F1-Score balanceado
   - **Falso negativo**: Risco de AVC a longo prazo
   - **Falso positivo**: Anticoagulação desnecessária

3. **NORM (Normal)**: 
   - **Métrica crítica**: Especificidade
   - **Falso negativo**: Paciente com condição não detectada
   - **Falso positivo**: Investigação desnecessária

### Thresholds Clínicos Recomendados
- **MI**: Recall > 90% (máxima sensibilidade)
- **AFIB**: F1-Score > 80% (balance precision/recall)
- **NORM**: Especificidade > 85% (evitar falsos alarmes)

---

## 🔗 Integração com Pipeline Completo

### Dependências de Entrada
- **Features processadas**: Output do Notebook II (CSV com features por canal)
- **Labels PTB-XL**: Arquivo ptbxl_database.csv
- **Estrutura de dados**: ECG ID como chave primária

### Outputs para Notebook IV
- **Modelo treinado**: Arquivo pickle para carregamento
- **Métricas detalhadas**: JSON estruturado para análise
- **Relatórios**: Texto legível para documentação

---

## 📦 Dependências

```python
import pandas as pd                          # >= 1.3.0
import numpy as np                           # >= 1.19.0
import pickle                                # Biblioteca padrão
import json                                  # Biblioteca padrão
from datetime import datetime                # Biblioteca padrão
import matplotlib.pyplot as plt              # >= 3.3.0
import seaborn as sns                        # >= 0.11.0

# Scikit-learn (versão >= 1.0.0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score, f1_score, cohen_kappa_score
```

---
