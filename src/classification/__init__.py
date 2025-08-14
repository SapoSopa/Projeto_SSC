import os
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Imports para machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, cohen_kappa_score
)

# Classes alvo do projeto
CLASSES_ALVO = {
    'MI': 'Infarto do Miocárdio',  
    'AFIB': 'Fibrilação Atrial',
    'NORM': 'Normal/Saudável'
}

# Mapeamento PTB-XL para nossas classes
MAPEAMENTO_PTB_XL = {
    'MI': ['IMI', 'ASMI', 'LMI', 'PMI', 'AMI'],   # Tipos de infarto
    'AFIB': ['AFIB', 'AFLT'],                     # Fibrilação atrial
    'NORM': ['NORM', 'SR']                        # Normal e ritmo sinusal
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

# Encontra o arquivo de características mais recente
def find_latest_features_file(data_path: str = "../data/") -> str:
    features_dir = os.path.join(data_path, "features")
    
    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Diretório não encontrado: {features_dir}")
    
    # Buscar arquivos CSV de features
    arquivos_features = [
        f for f in os.listdir(features_dir) 
        if f.endswith('.csv') and 'features' in f.lower()
    ]
    
    if not arquivos_features:
        raise FileNotFoundError(f"Nenhum arquivo de features encontrado em {features_dir}")
    
    # Pegar o mais recente (por nome, assumindo formato com timestamp)
    arquivo_mais_recente = sorted(arquivos_features)[-1]
    
    return os.path.join(features_dir, arquivo_mais_recente)

# Encontra o arquivo de banco de dados PTB-XL
def find_ptbxl_database(data_path: str = "../data/") -> str:
    # Possíveis localizações
    possible_paths = [
        os.path.join(data_path, "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1", "ptbxl_database.csv"),
        os.path.join(data_path, "raw", "ptbxl_database.csv"),
        os.path.join(data_path, "ptbxl_database.csv"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Arquivo PTB-XL não encontrado. Tentativas: {possible_paths}")

# Determina a classe principal baseada nos códigos SCP
def determinar_classe_principal(scp_codes: str, mapeamento: Dict[str, List[str]]) -> str:
    if pd.isna(scp_codes) or scp_codes == '{}':
        return None
    
    try:
        # Converter string para dict
        if isinstance(scp_codes, str):
            codes_dict = eval(scp_codes)
        else:
            codes_dict = scp_codes
        
        # Verificar cada classe por prioridade
        for classe, codigos in mapeamento.items():
            for codigo in codigos:
                if codigo in codes_dict:
                    return classe
        
        return None
        
    except:
        return None

# Carrega o dataset de características extraídas
def load_features_dataset(filepath: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    print(f" Carregando dataset de características:")
    
    try:
        if filepath is None:
            filepath = find_latest_features_file()
        
        print(f"   Arquivo: {filepath}")
        
        # Carregar CSV
        df = pd.read_csv(filepath)
        
        # Detectar metadados
        metadata = {
            'filepath': filepath,
            'total_records': len(df),
            'unique_ecgs': df['ecg_id'].nunique() if 'ecg_id' in df.columns else 0,
            'columns': list(df.columns)
        }
        
        # Detectar coluna de canal
        channel_columns = ['canal', 'channel', 'lead', 'derivacao']
        for col in channel_columns:
            if col in df.columns:
                metadata['channel_column'] = col
                metadata['channels_per_ecg'] = df.groupby('ecg_id').size().iloc[0] if 'ecg_id' in df.columns else 0
                break
        
        print(f"   Dataset carregado: {df.shape}")
        return df, metadata
        
    except Exception as e:
        raise FileNotFoundError(f"Erro ao carregar features: {str(e)}")

# Carrega e processa os diagnósticos da base PTB-XL
def load_labels_ptbxl(ptbxl_path: str = None, ecg_ids: List[int] = None) -> pd.DataFrame:
    print(f" Carregando diagnósticos PTB-XL:")
    
    try:
        if ptbxl_path is None:
            ptbxl_path = find_ptbxl_database()
        
        print(f"   Arquivo PTB-XL: {ptbxl_path}")
        
        # Carregar dados PTB-XL
        df_ptbxl = pd.read_csv(ptbxl_path)
        print(f"   Registros PTB-XL: {len(df_ptbxl)}")
        
        # Filtrar apenas os ECGs que temos
        if ecg_ids:
            df_ptbxl = df_ptbxl[df_ptbxl['ecg_id'].isin(ecg_ids)]
            print(f"   Filtrado para {len(df_ptbxl)} ECGs disponíveis")
        
        # Mapear diagnósticos para nossas classes
        df_ptbxl['classe'] = df_ptbxl['scp_codes'].apply(
            lambda x: determinar_classe_principal(x, MAPEAMENTO_PTB_XL)
        )
        
        # Remover casos sem classe definida
        df_labels = df_ptbxl[df_ptbxl['classe'].notna()].copy()
        
        print(f"   Labels mapeados: {len(df_labels)} ECGs com diagnóstico")
        
        # Mostrar distribuição
        distribuicao = df_labels['classe'].value_counts()
        print(f"\n   DISTRIBUIÇÃO DAS CLASSES:")
        for classe, count in distribuicao.items():
            nome_completo = CLASSES_ALVO[classe]
            pct = 100 * count / len(df_labels)
            print(f"     {classe} ({nome_completo}): {count} ({pct:.1f}%)")
        
        return df_labels[['ecg_id', 'classe']].reset_index(drop=True)
        
    except Exception as e:
        raise FileNotFoundError(f"Erro ao carregar labels PTB-XL: {str(e)}")

#   Converte dados por canal para dados por paciente
def preparar_dados_por_paciente(df_features: pd.DataFrame) -> pd.DataFrame:
    print(f" Dados por paciente:")
    
    # Verificar se existe coluna ecg_id
    if 'ecg_id' not in df_features.columns:
        raise ValueError("Coluna 'ecg_id' não encontrada no dataset")
    
    # Detectar coluna de canal
    channel_column = None
    for col in ['canal', 'channel', 'lead', 'derivacao']:
        if col in df_features.columns:
            channel_column = col
            print(f"   Coluna de canal : '{col}'")
            break
    
    # Colunas a serem removidas (auxiliares)
    colunas_descartar = []
    for col in ['canal', 'channel', 'lead', 'derivacao', 'fs', 'n_samples', 'arquivo_origem', 'filename']:
        if col in df_features.columns:
            colunas_descartar.append(col)
    
    # Remover colunas auxiliares se existirem
    if colunas_descartar:
        df_limpo = df_features.drop(columns=colunas_descartar)
        print(f"   Colunas auxiliares removidas: {colunas_descartar}")
    else:
        df_limpo = df_features.copy()
        print(f"   Nenhuma coluna auxiliar para remover")
    
    print(f"   Colunas restantes: {list(df_limpo.columns)}")
    
    # Verificar se temos dados por canal ou já estão agrupados
    unique_ecgs = df_limpo['ecg_id'].nunique()
    total_records = len(df_limpo)
    
    if total_records == unique_ecgs:
        print(f"   Dados já estão por paciente (1 registro por ECG)")
        return df_limpo
    else:
        print(f"   Convertendo {total_records} registros para {unique_ecgs} pacientes")
    
    # Agrupar por paciente concatenando features de todos os canais
    dados_pacientes = []
    
    for ecg_id in df_limpo['ecg_id'].unique():
        dados_ecg = df_limpo[df_limpo['ecg_id'] == ecg_id]
        
        # Remover coluna ecg_id temporariamente para concatenar apenas features
        features_canais = dados_ecg.drop(columns=['ecg_id']).values
        
        # Concatenar todas as features de todos os canais em um vetor único
        features_concatenadas = features_canais.flatten()
        
        dados_pacientes.append([ecg_id, features_concatenadas])
    
    # Criar DataFrame organizado
    df_pacientes = pd.DataFrame(dados_pacientes, columns=['ecg_id', 'features'])
    
    print(f" Conversão concluída:")
    print(f"   Pacientes: {len(dados_pacientes)}")
    print(f"   Features por paciente: {len(dados_pacientes[0][1]) if dados_pacientes else 0}")
    
    return df_pacientes

# Classificador Random Forest especializado para ECG
class ECGRandomForestClassifier:
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or RF_CONFIG_DEFAULT
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_history = {}
        
    # Prepara dados para treinamento
    def prepare_data(self, df_features: pd.DataFrame, df_labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # Fazer merge dos dados
        df_merged = df_features.merge(df_labels, on='ecg_id', how='inner')
        print(f"   ECGs com features e labels: {len(df_merged)}")
        
        # Extrair features
        X = np.vstack(df_merged['features'].values)
        
        # Extrair labels
        y = self.label_encoder.fit_transform(df_merged['classe'].values)
        
        # Normalizar features
        X = self.scaler.fit_transform(X)
        
        print(f"   Dados preparados: X={X.shape}, y={y.shape}")
        print(f"   Classes mapeadas: {dict(enumerate(self.label_encoder.classes_))}")
        
        return X, y
    
    # Divide dados em treino e teste
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    # Treina o modelo
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              optimize_hyperparams: bool = True, cv_folds: int = 5) -> Dict[str, Any]:
        print("Iniciando treinamento:")
        start_time = datetime.now()
        
        if optimize_hyperparams:
            print(" Otimizando hiperparâmetros com Grid Search...")
            
            # Grid Search com validação cruzada
            grid_search = GridSearchCV(
                RandomForestClassifier(**self.config),
                RF_PARAM_GRID,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            print(f" Melhores parâmetros: {grid_search.best_params_}")
            print(f" Melhor score CV: {grid_search.best_score_:.4f}")
            
            # Salvar resultados
            self.training_history = {
                'best_params': grid_search.best_params_,
                'cv_mean': grid_search.best_score_,
                'cv_std': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                'n_features': X_train.shape[1],
                'training_time': (datetime.now() - start_time).total_seconds()
            }
            
        else:
            print(" Treinando com parâmetros padrão...")
            self.model = RandomForestClassifier(**self.config)
            
            # Validação cruzada
            cv_scores = cross_val_score(self.model, X_train, y_train, 
                                      cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                                      scoring='f1_weighted')
            
            self.model.fit(X_train, y_train)
            
            self.training_history = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_features': X_train.shape[1],
                'training_time': (datetime.now() - start_time).total_seconds()
            }
        
        print(f" Tempo de treinamento: {self.training_history['training_time']:.1f}s")
        return self.training_history
    
    # Realiza predições
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        return self.model.predict(X)
    
    # Retorna probabilidades das predições
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        probas = self.model.predict_proba(X)
        return {
            'probabilities': probas,
            'classes': self.label_encoder.classes_
        }
    
    # Avalia o modelo no conjunto de teste
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        print(" Avaliando modelo no conjunto de teste...")
        
        # Predições
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=self.label_encoder.classes_,
                                                         output_dict=True),
            'y_true': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba['probabilities']
        }
        
        print(f" Avaliação concluída:")
        print(f"    Acurácia: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"    Cohen's Kappa (κ): {metrics['cohen_kappa']:.4f}")
        
        return metrics

# Plota resultados da classificação
def plot_classification_results(metrics: Dict[str, Any], 
                               save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Resultados da Classificação ECG - Random Forest', fontsize=16, fontweight='bold')
    
    # 1. Matriz de confusão
    cm = metrics['confusion_matrix']
    classes = list(CLASSES_ALVO.keys())
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                ax=axes[0, 0])
    axes[0, 0].set_title('Matriz de Confusão')
    axes[0, 0].set_xlabel('Predito')
    axes[0, 0].set_ylabel('Real')
    
    # 2. Métricas por classe
    report = metrics['classification_report']
    classes_metrics = [report[classe] for classe in classes if classe in report]
    
    metrics_names = ['precision', 'recall', 'f1-score']
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics_names):
        values = [m[metric] for m in classes_metrics]
        axes[0, 1].bar(x + i*width, values, width, label=metric.capitalize())
    
    axes[0, 1].set_title('Métricas por Classe')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(classes)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuição de probabilidades
    y_proba = metrics['y_proba']
    for i, classe in enumerate(classes):
        axes[1, 0].hist(y_proba[:, i], bins=20, alpha=0.7, label=f'Classe {classe}')
    
    axes[1, 0].set_title('Distribuição de Probabilidades')
    axes[1, 0].set_xlabel('Probabilidade')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Métricas gerais
    general_metrics = {
        'Acurácia': metrics['accuracy'],
        'Acurácia Balanceada': metrics['balanced_accuracy'],
        'F1-Score Weighted': metrics['f1_weighted'],
        'Cohen Kappa': metrics['cohen_kappa']
    }
    
    bars = axes[1, 1].bar(general_metrics.keys(), general_metrics.values(),
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1, 1].set_title('Métricas Gerais')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Adicionar valores nas barras
    for bar, value in zip(bars, general_metrics.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Plota importância das características
def plot_feature_importance(importance_dict: Dict[str, float], 
                          save_path: Optional[str] = None) -> None:
    # Ordenar por importância
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Pegar top 20
    top_features = sorted_features[:20]
    
    features, importance = zip(*top_features)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(features)), importance, color='skyblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importância')
    plt.title('Top 20 Características Mais Importantes - Random Forest')
    plt.gca().invert_yaxis()
    
    # Adicionar valores nas barras
    for i, (bar, imp) in enumerate(zip(bars, importance)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', ha='left', va='center')
    
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Salva o modelo treinado e resultados
def save_model_and_results(classifier: ECGRandomForestClassifier, 
                          metrics: Dict[str, Any],
                          output_dir: str = "../results/classification") -> Dict[str, str]:
    print(f" Salvando modelo e resultados...")
    
    # Criar diretório
    os.makedirs(output_dir, exist_ok=True)
     
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Salvar modelo completo
    model_path = f"{output_dir}/ecg_random_forest_model_{timestamp}.pkl"
    
    model_data = {
        'classifier': classifier,
        'training_history': classifier.training_history,
        'metadata': {
            'timestamp': timestamp,
            'classes': list(CLASSES_ALVO.keys()),
            'model_type': 'RandomForest',
            'n_features': classifier.training_history.get('n_features', 0)
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"   Modelo salvo: {model_path}")
    
    # 2. Salvar métricas em JSON
    metrics_path = f"{output_dir}/evaluation_metrics_{timestamp}.json"
    
    # Converter arrays numpy para listas para JSON
    metrics_json = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_json[key] = value.tolist()
        elif key == 'classification_report':
            metrics_json[key] = value  # Já é dict serializable
        else:
            metrics_json[key] = value
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)
    
    print(f"   Métricas salvas: {metrics_path}")
    
    # 3. Relatório texto
    report_path = f"{output_dir}/classification_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("RELATÓRIO DE CLASSIFICAÇÃO ECG - RANDOM FOREST\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Classes: {list(CLASSES_ALVO.keys())}\n")
        f.write(f"Acurácia: {metrics['accuracy']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}\n\n")
        
        f.write("MATRIZ DE CONFUSÃO:\n")
        f.write(str(metrics['confusion_matrix']) + "\n\n")
        
        f.write("RELATÓRIO DETALHADO:\n")
        from sklearn.metrics import classification_report
        report_str = classification_report(
            metrics['y_true'], 
            metrics['y_pred'], 
            target_names=classifier.label_encoder.classes_
        )
        f.write(report_str)
    
    print(f"   Relatório salvo: {report_path}")
    
    return {
        'model_path': model_path,
        'metrics_path': metrics_path,
        'report_path': report_path
    }

# Exportar símbolos principais
__all__ = [
    'ECGRandomForestClassifier',
    'load_features_dataset',
    'load_labels_ptbxl',
    'preparar_dados_por_paciente',
    'determinar_classe_principal',
    'plot_classification_results',
    'plot_feature_importance',
    'save_model_and_results',
    'CLASSES_ALVO',
    'MAPEAMENTO_PTB_XL',
    'RF_CONFIG_DEFAULT',
    'RF_PARAM_GRID'
]