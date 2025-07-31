"""
Módulo de classificação
Contém implementações de diferentes algoritmos de classificação
"""
#Criado no chat
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

class SignalClassifier:
    """
    Classe base para classificação de sinais
    """
    
    def __init__(self, classifier_type: str = 'random_forest'):
        """
        Inicializa o classificador
        
        Args:
            classifier_type: Tipo de classificador ('random_forest', 'svm', 'knn', 'naive_bayes')
        """
        self.classifier_type = classifier_type
        self.model = self._get_classifier(classifier_type)
        self.is_trained = False
        
    def _get_classifier(self, classifier_type: str):
        """Retorna o modelo de classificação especificado"""
        classifiers = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(kernel='rbf', random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB()
        }
        
        if classifier_type not in classifiers:
            raise ValueError(f"Classificador {classifier_type} não suportado")
            
        return classifiers[classifier_type]
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Treina o classificador
        
        Args:
            X: Características de entrada
            y: Labels/classes
        """
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições
        
        Args:
            X: Características de entrada
            
        Returns:
            Predições
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
            
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Avalia o desempenho do classificador
        
        Args:
            X_test: Características de teste
            y_test: Labels de teste
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if not self.is_trained:
            raise ValueError("Modelo precisa ser treinado primeiro")
            
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics

def compare_classifiers(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Compara diferentes algoritmos de classificação
    
    Args:
        X: Características
        y: Labels
        test_size: Proporção dos dados para teste
        
    Returns:
        Dicionário com métricas para cada classificador
    """
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    classifiers = ['random_forest', 'svm', 'knn', 'naive_bayes']
    results = {}
    
    for clf_name in classifiers:
        try:
            classifier = SignalClassifier(clf_name)
            classifier.train(X_train, y_train)
            metrics = classifier.evaluate(X_test, y_test)
            results[clf_name] = metrics
        except Exception as e:
            print(f"Erro ao treinar {clf_name}: {e}")
            results[clf_name] = None
    
    return results
