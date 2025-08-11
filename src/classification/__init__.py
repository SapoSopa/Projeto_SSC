# """
# Módulo de classificação
# Contém implementações de diferentes algoritmos de classificação
# """
# #Criado no chat
# import numpy as np
# from typing import Tuple, Dict, Any, Optional
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB

# class SignalClassifier:
#     """
#     Classe base para classificação de sinais
#     """
    
#     def __init__(self, classifier_type: str = 'random_forest'):
#         """
#         Inicializa o classificador
        
#         Args:
#             classifier_type: Tipo de classificador ('random_forest', 'svm', 'knn', 'naive_bayes')
#         """
#         self.classifier_type = classifier_type
#         self.model = self._get_classifier(classifier_type)
#         self.is_trained = False
        
#     def _get_classifier(self, classifier_type: str):
#         """Retorna o modelo de classificação especificado"""
#         classifiers = {
#             'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
#             'svm': SVC(kernel='rbf', random_state=42),
#             'knn': KNeighborsClassifier(n_neighbors=5),
#             'naive_bayes': GaussianNB()
#         }
        
#         if classifier_type not in classifiers:
#             raise ValueError(f"Classificador {classifier_type} não suportado")
            
#         return classifiers[classifier_type]
    
#     def train(self, X: np.ndarray, y: np.ndarray) -> None:
#         """
#         Treina o classificador
        
#         Args:
#             X: Características de entrada
#             y: Labels/classes
#         """
#         self.model.fit(X, y)
#         self.is_trained = True
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Faz predições
        
#         Args:
#             X: Características de entrada
            
#         Returns:
#             Predições
#         """
#         if not self.is_trained:
#             raise ValueError("Modelo precisa ser treinado primeiro")
            
#         return self.model.predict(X)
    
#     def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
#         """
#         Avalia o desempenho do classificador
        
#         Args:
#             X_test: Características de teste
#             y_test: Labels de teste
            
#         Returns:
#             Dicionário com métricas de avaliação
#         """
#         if not self.is_trained:
#             raise ValueError("Modelo precisa ser treinado primeiro")
            
#         y_pred = self.predict(X_test)
        
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted'),
#             'recall': recall_score(y_test, y_pred, average='weighted'),
#             'f1_score': f1_score(y_test, y_pred, average='weighted')
#         }
        
#         return metrics

"""
Módulo de Classificação de Sinais ECG
Contém implementações especializadas para classificação de arritmias cardíacas
Otimizado para dataset PTB-XL com foco em 3 classes principais
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Imports do scikit-learn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score
)

# Imports para visualização
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ============================================================================

# Classes alvo do projeto
CLASSES_ALVO = {
    'MI': 'Infarto do Miocárdio',      # Myocardial Infarction  
    'AFIB': 'Fibrilação Atrial',       # Atrial Fibrillation
    'NORM': 'Saudável'                 # Normal/Healthy
}

# Mapeamento PTB-XL para nossas classes
MAPEAMENTO_PTB_XL = {
    'MI': ['IMI', 'ASMI', 'LMI', 'PMI'],  # Tipos de infarto
    'AFIB': ['AFIB'],                      # Fibrilação atrial
    'NORM': ['NORM']                       # Normal
}

# Configuração padrão do Random Forest
RF_CONFIG_DEFAULT = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Grid de hiperparâmetros para otimização
RF_PARAM_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# ============================================================================
# FUNÇÕES DE CARREGAMENTO E PREPARAÇÃO DE DADOS
# ============================================================================

def load_features_dataset(filepath: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carrega o dataset de características extraído no Notebook II
    
    Args:
        filepath: Caminho para o arquivo CSV de características
        
    Returns:
        Tuple contendo DataFrame e metadados
    """
    print(f"📂 Carregando dataset de características: {filepath}")
    
    try:
        # Carregar dataset
        df = pd.read_csv(filepath)
        
        # Verificar estrutura
        print(f"   Shape: {df.shape}")
        print(f"   ECGs únicos: {df['ecg_id'].nunique()}")
        print(f"   Canais por ECG: {df.groupby('ecg_id').size().iloc[0]}")
        
        # Metadados básicos
        metadata = {
            'total_registros': len(df),
            'total_ecgs': df['ecg_id'].nunique(),
            'canais_por_ecg': df.groupby('ecg_id').size().iloc[0] if len(df) > 0 else 0,
            'features_disponiveis': [col for col in df.columns 
                                   if col not in ['ecg_id', 'canal', 'fs', 'n_samples', 'arquivo_origem']],
            'timestamp_carregamento': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        print(f"   Features disponíveis: {len(metadata['features_disponiveis'])}")
        return df, metadata
        
    except Exception as e:
        raise FileNotFoundError(f"Erro ao carregar dataset: {str(e)}")

def load_labels_ptbxl(ptbxl_path: str, ecg_ids: List[int]) -> pd.DataFrame:
    """
    Carrega e processa os labels do PTB-XL para as classes alvo
    
    Args:
        ptbxl_path: Caminho para o arquivo ptbxl_database.csv
        ecg_ids: Lista de IDs dos ECGs para carregar labels
        
    Returns:
        DataFrame com labels processados
    """
    print(f"🏷️ Carregando labels PTB-XL: {ptbxl_path}")
    
    try:
        # Carregar base PTB-XL
        ptbxl_df = pd.read_csv(ptbxl_path)
        
        # Filtrar apenas ECGs de interesse
        ptbxl_filtered = ptbxl_df[ptbxl_df['ecg_id'].isin(ecg_ids)].copy()
        
        print(f"   ECGs encontrados: {len(ptbxl_filtered)}/{len(ecg_ids)}")
        
        # Processar diagnósticos
        labels_processados = []
        
        for _, row in ptbxl_filtered.iterrows():
            ecg_id = row['ecg_id']
            scp_codes = eval(row['scp_codes']) if isinstance(row['scp_codes'], str) else {}
            
            # Determinar classe principal
            classe = determinar_classe_principal(scp_codes)
            
            labels_processados.append({
                'ecg_id': ecg_id,
                'classe': classe,
                'scp_codes_original': scp_codes,
                'age': row.get('age', np.nan),
                'sex': row.get('sex', np.nan)
            })
        
        df_labels = pd.DataFrame(labels_processados)
        
        # Estatísticas das classes
        print(f"   Distribuição de classes:")
        for classe, count in df_labels['classe'].value_counts().items():
            nome_completo = CLASSES_ALVO.get(classe, classe)
            pct = 100 * count / len(df_labels)
            print(f"     {classe} ({nome_completo}): {count} ({pct:.1f}%)")
        
        return df_labels
        
    except Exception as e:
        raise FileNotFoundError(f"Erro ao carregar labels PTB-XL: {str(e)}")

def determinar_classe_principal(scp_codes: Dict[str, float]) -> str:
    """
    Determina a classe principal baseada nos códigos SCP
    
    Args:
        scp_codes: Dicionário com códigos SCP e probabilidades
        
    Returns:
        Classe principal ('MI', 'AFIB', ou 'NORM')
    """
    # Verificar cada classe em ordem de prioridade
    
    # 1. Infarto do Miocárdio (prioridade alta)
    for codigo in MAPEAMENTO_PTB_XL['MI']:
        if codigo in scp_codes and scp_codes[codigo] > 0:
            return 'MI'
    
    # 2. Fibrilação Atrial
    for codigo in MAPEAMENTO_PTB_XL['AFIB']:
        if codigo in scp_codes and scp_codes[codigo] > 0:
            return 'AFIB'
    
    # 3. Normal (padrão)
    for codigo in MAPEAMENTO_PTB_XL['NORM']:
        if codigo in scp_codes and scp_codes[codigo] > 0:
            return 'NORM'
    
    # Se não encontrou nenhuma, classificar como Normal
    return 'NORM'

def preparar_dados_por_paciente(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Converte dados por canal para dados por paciente (concatenando todas as derivações)
    
    Args:
        df_features: DataFrame com características por canal
        
    Returns:
        DataFrame com características concatenadas por paciente
    """
    print("🔄 Preparando dados por paciente...")
    
    # Identificar colunas de características
    colunas_features = [col for col in df_features.columns 
                       if col not in ['ecg_id', 'canal', 'fs', 'n_samples', 'arquivo_origem']]
    
    # Agrupar por ECG e concatenar características
    dados_pacientes = []
    
    for ecg_id, grupo in df_features.groupby('ecg_id'):
        # Ordenar por canal para manter ordem consistente
        grupo_ordenado = grupo.sort_values('canal')
        
        # Concatenar características de todos os canais
        features_concatenadas = grupo_ordenado[colunas_features].values.flatten()
        
        dados_pacientes.append({
            'ecg_id': ecg_id,
            'features': features_concatenadas
        })
    
    df_pacientes = pd.DataFrame(dados_pacientes)
    
    print(f"   ECGs processados: {len(df_pacientes)}")
    print(f"   Features por paciente: {len(df_pacientes.iloc[0]['features'])}")
    
    return df_pacientes

# ============================================================================
# CLASSE PRINCIPAL DE CLASSIFICAÇÃO
# ============================================================================

class ECGRandomForestClassifier:
    """
    Classificador Random Forest especializado para ECG
    Otimizado para classificação de 3 classes: MI, AFIB, NORM
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializa o classificador
        
        Args:
            config: Configurações personalizadas do Random Forest
        """
        self.config = config or RF_CONFIG_DEFAULT.copy()
        self.model = RandomForestClassifier(**self.config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Estado do modelo
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        self.classes = list(CLASSES_ALVO.keys())
        
        print(f"🌲 Random Forest Classifier inicializado")
        print(f"   Configuração: {self.config}")
    
    def prepare_data(self, df_features: pd.DataFrame, df_labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara os dados para treinamento
        
        Args:
            df_features: DataFrame com características por paciente
            df_labels: DataFrame com labels
            
        Returns:
            Tuple (X, y) com dados preparados
        """
        print("🔧 Preparando dados para classificação...")
        
        # Fazer merge dos dados
        df_merged = df_features.merge(df_labels, on='ecg_id', how='inner')
        
        print(f"   Registros após merge: {len(df_merged)}")
        
        # Extrair características e labels
        X = np.array(df_merged['features'].tolist())
        y = df_merged['classe'].values
        
        # Verificar balanceamento
        print(f"   Distribuição de classes:")
        for classe in np.unique(y):
            count = np.sum(y == classe)
            pct = 100 * count / len(y)
            nome_completo = CLASSES_ALVO.get(classe, classe)
            print(f"     {classe} ({nome_completo}): {count} ({pct:.1f}%)")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide os dados em treino e teste
        
        Args:
            X: Características
            y: Labels
            test_size: Proporção do conjunto de teste
            random_state: Semente aleatória
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        print(f"📊 Dividindo dados (treino: {100*(1-test_size):.0f}%, teste: {100*test_size:.0f}%)...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y  # Manter proporção de classes
        )
        
        print(f"   Treino: {len(X_train)} amostras")
        print(f"   Teste: {len(X_test)} amostras")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              optimize_hyperparams: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Treina o modelo Random Forest
        
        Args:
            X_train: Características de treino
            y_train: Labels de treino
            optimize_hyperparams: Se deve otimizar hiperparâmetros
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            Dicionário com resultados do treinamento
        """
        print(f"🎯 Iniciando treinamento do Random Forest...")
        start_time = datetime.now()
        
        # Normalizar dados
        print("   Normalizando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Codificar labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if optimize_hyperparams:
            print(f"   Otimizando hiperparâmetros com GridSearchCV ({cv_folds} folds)...")
            
            # Grid Search
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid=RF_PARAM_GRID,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train_scaled, y_train_encoded)
            
            # Usar melhor modelo
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = grid_search.best_score_
            
            print(f"   Melhores parâmetros: {best_params}")
            print(f"   Melhor CV Score: {best_cv_score:.4f}")
            
        else:
            print("   Treinando com parâmetros padrão...")
            self.model.fit(X_train_scaled, y_train_encoded)
            best_params = self.config
            best_cv_score = None
        
        # Validação cruzada final
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train_encoded,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='f1_weighted'
        )
        
        # Marcar como treinado
        self.is_trained = True
        
        # Salvar histórico
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        self.training_history = {
            'timestamp': start_time.isoformat(),
            'training_time_seconds': training_time,
            'optimize_hyperparams': optimize_hyperparams,
            'best_params': best_params,
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples_train': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        print(f"✅ Treinamento concluído em {training_time:.2f}s")
        print(f"   CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Características de entrada
            
        Returns:
            Predições decodificadas
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Faz predições probabilísticas
        
        Args:
            X: Características de entrada
            
        Returns:
            Dicionário com probabilidades por classe
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
        
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)
        
        # Mapear para nomes das classes
        classes = self.label_encoder.classes_
        proba_dict = {}
        
        for i, classe in enumerate(classes):
            proba_dict[classe] = proba[:, i]
        
        return proba_dict
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test: Características de teste
            y_test: Labels de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        print("📈 Avaliando modelo no conjunto de teste...")
        
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
        
        # Fazer predições
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # Codificar para métricas
        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test_encoded, y_pred_encoded).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'n_samples_test': len(y_test)
        }
        
        # Métricas por classe
        for classe in self.label_encoder.classes_:
            y_binary = (y_test == classe).astype(int)
            y_pred_binary = (y_pred == classe).astype(int)
            
            if len(np.unique(y_binary)) > 1:  # Só calcular se a classe existe no teste
                metrics[f'f1_{classe}'] = f1_score(y_binary, y_pred_binary)
                metrics[f'precision_{classe}'] = precision_score(y_binary, y_pred_binary)
                metrics[f'recall_{classe}'] = recall_score(y_binary, y_pred_binary)
        
        print(f"   Acurácia: {metrics['accuracy']:.4f}")
        print(f"   F1-Score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"   Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Obtém importância das características
        
        Args:
            top_n: Número de características mais importantes
            
        Returns:
            Dicionário com importâncias
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
        
        importances = self.model.feature_importances_
        
        # Criar nomes das características (se não foram fornecidos)
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        # Ordenar por importância
        indices = np.argsort(importances)[::-1]
        
        importance_dict = {}
        for i in range(min(top_n, len(importances))):
            idx = indices[i]
            importance_dict[self.feature_names[idx]] = importances[idx]
        
        return importance_dict

# ============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================================

def plot_classification_results(metrics: Dict[str, Any], 
                               save_path: Optional[str] = None) -> None:
    """
    Visualiza resultados da classificação
    
    Args:
        metrics: Dicionário com métricas de avaliação
        save_path: Caminho para salvar a figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('🎯 Resultados da Classificação ECG - Random Forest', 
                 fontsize=16, fontweight='bold')
    
    # 1. Matriz de Confusão
    ax1 = axes[0, 0]
    cm = np.array(metrics['confusion_matrix'])
    classes = list(CLASSES_ALVO.keys())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=ax1, cbar_kws={'label': 'Número de Amostras'})
    ax1.set_title('Matriz de Confusão', fontweight='bold')
    ax1.set_xlabel('Classe Predita')
    ax1.set_ylabel('Classe Real')
    
    # 2. Métricas por Classe
    ax2 = axes[0, 1]
    report = metrics['classification_report']
    
    metricas_classes = []
    for classe in classes:
        if classe in report:
            metricas_classes.append([
                report[classe]['precision'],
                report[classe]['recall'],
                report[classe]['f1-score']
            ])
    
    if metricas_classes:
        x = np.arange(len(classes))
        width = 0.25
        
        precision = [m[0] for m in metricas_classes]
        recall = [m[1] for m in metricas_classes]
        f1 = [m[2] for m in metricas_classes]
        
        ax2.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax2.bar(x, recall, width, label='Recall', alpha=0.8)
        ax2.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax2.set_title('Métricas por Classe', fontweight='bold')
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{c}\n({CLASSES_ALVO[c]})" for c in classes])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Métricas Globais
    ax3 = axes[1, 0]
    metricas_globais = {
        'Acurácia': metrics['accuracy'],
        'Acurácia Balanceada': metrics['balanced_accuracy'],
        'F1 Weighted': metrics['f1_weighted'],
        'Cohen Kappa': metrics['cohen_kappa']
    }
    
    bars = ax3.bar(metricas_globais.keys(), metricas_globais.values(), 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
    ax3.set_title('Métricas Globais', fontweight='bold')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, valor in zip(bars, metricas_globais.values()):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{valor:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribuição de Classes
    ax4 = axes[1, 1]
    
    # Contar predições por classe (se disponível)
    if 'y_pred' in metrics:
        unique, counts = np.unique(metrics['y_pred'], return_counts=True)
        counts_dict = dict(zip(unique, counts))
    else:
        # Usar dados da matriz de confusão
        counts_dict = {classes[i]: cm[i, :].sum() for i in range(len(classes))}
    
    wedges, texts, autotexts = ax4.pie(counts_dict.values(), 
                                       labels=[f"{k}\n({CLASSES_ALVO[k]})" for k in counts_dict.keys()],
                                       autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribuição de Classes', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Figura salva em: {save_path}")
    
    plt.show()

def plot_feature_importance(importance_dict: Dict[str, float], 
                          save_path: Optional[str] = None) -> None:
    """
    Visualiza importância das características
    
    Args:
        importance_dict: Dicionário com importâncias
        save_path: Caminho para salvar a figura
    """
    plt.figure(figsize=(12, 8))
    
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    
    # Criar gráfico de barras horizontais
    y_pos = np.arange(len(features))
    plt.barh(y_pos, importances, alpha=0.8, color='steelblue')
    
    plt.yticks(y_pos, features)
    plt.xlabel('Importância')
    plt.title('🔍 Importância das Características - Random Forest', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Adicionar valores nas barras
    for i, v in enumerate(importances):
        plt.text(v + max(importances)*0.01, i, f'{v:.4f}', 
                va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Figura salva em: {save_path}")
    
    plt.show()

# ============================================================================
# FUNÇÕES DE SALVAMENTO E RELATÓRIOS
# ============================================================================

def save_model_and_results(classifier: ECGRandomForestClassifier, 
                          metrics: Dict[str, Any],
                          output_dir: str = "../results/classification") -> Dict[str, str]:
    """
    Salva o modelo treinado e resultados
    
    Args:
        classifier: Modelo treinado
        metrics: Métricas de avaliação
        output_dir: Diretório de saída
        
    Returns:
        Dicionário com caminhos dos arquivos salvos
    """
    import pickle
    
    # Criar diretório
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Salvar modelo
    model_path = f"{output_dir}/random_forest_model_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    # Salvar métricas
    metrics_path = f"{output_dir}/classification_metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        # Converter arrays numpy para listas para JSON
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            else:
                metrics_json[key] = value
        
        json.dump(metrics_json, f, indent=2, default=str)
    
    # Criar relatório resumido
    report_path = f"{output_dir}/classification_report_{timestamp}.md"
    with open(report_path, 'w') as f:
        f.write(f"# Relatório de Classificação ECG - Random Forest\n\n")
        f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        f.write(f"## Configuração do Modelo\n")
        f.write(f"- **Algoritmo:** Random Forest\n")
        f.write(f"- **Classes:** {', '.join(CLASSES_ALVO.keys())}\n")
        f.write(f"- **Parâmetros:** {classifier.training_history.get('best_params', 'N/A')}\n\n")
        f.write(f"## Resultados Principais\n")
        f.write(f"- **Acurácia:** {metrics['accuracy']:.4f}\n")
        f.write(f"- **F1-Score (weighted):** {metrics['f1_weighted']:.4f}\n")
        f.write(f"- **Cohen's Kappa:** {metrics['cohen_kappa']:.4f}\n")
        f.write(f"- **Amostras de teste:** {metrics['n_samples_test']}\n\n")
        f.write(f"## Tempo de Treinamento\n")
        f.write(f"- **Duração:** {classifier.training_history.get('training_time_seconds', 0):.2f}s\n")
        f.write(f"- **Validação Cruzada:** {classifier.training_history.get('cv_mean', 0):.4f} ± {classifier.training_history.get('cv_std', 0):.4f}\n")
    
    arquivos_salvos = {
        'modelo': model_path,
        'metricas': metrics_path,
        'relatorio': report_path
    }
    
    print(f"💾 Arquivos salvos:")
    for tipo, caminho in arquivos_salvos.items():
        print(f"   {tipo.capitalize()}: {caminho}")
    
    return arquivos_salvos

# ============================================================================
# FUNÇÃO PRINCIPAL DE PIPELINE
# ============================================================================

def run_classification_pipeline(features_path: str, 
                              ptbxl_path: str,
                              output_dir: str = "../results/classification",
                              optimize_hyperparams: bool = True) -> Dict[str, Any]:
    """
    Executa o pipeline completo de classificação
    
    Args:
        features_path: Caminho para o dataset de características
        ptbxl_path: Caminho para o arquivo PTB-XL
        output_dir: Diretório de saída
        optimize_hyperparams: Se deve otimizar hiperparâmetros
        
    Returns:
        Dicionário com resultados completos
    """
    print("🚀 INICIANDO PIPELINE DE CLASSIFICAÇÃO ECG")
    print("="*60)
    
    # 1. Carregar dados
    df_features, features_metadata = load_features_dataset(features_path)
    
    # 2. Preparar dados por paciente
    df_pacientes = preparar_dados_por_paciente(df_features)
    
    # 3. Carregar labels
    ecg_ids = df_pacientes['ecg_id'].tolist()
    df_labels = load_labels_ptbxl(ptbxl_path, ecg_ids)
    
    # 4. Inicializar classificador
    classifier = ECGRandomForestClassifier()
    
    # 5. Preparar dados
    X, y = classifier.prepare_data(df_pacientes, df_labels)
    
    # 6. Dividir dados
    X_train, X_test, y_train, y_test = classifier.split_data(X, y)
    
    # 7. Treinar modelo
    training_results = classifier.train(X_train, y_train, optimize_hyperparams=optimize_hyperparams)
    
    # 8. Avaliar modelo
    metrics = classifier.evaluate(X_test, y_test)
    
    # 9. Salvar resultados
    arquivos = save_model_and_results(classifier, metrics, output_dir)
    
    # 10. Visualizar resultados
    plot_classification_results(metrics, f"{output_dir}/classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # 11. Importância das características
    importance = classifier.get_feature_importance()
    plot_feature_importance(importance, f"{output_dir}/feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    print("✅ PIPELINE DE CLASSIFICAÇÃO CONCLUÍDO!")
    print("="*60)
    
    return {
        'classifier': classifier,
        'metrics': metrics,
        'training_results': training_results,
        'feature_importance': importance,
        'files_saved': arquivos
    }