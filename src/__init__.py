"""
Projeto de Sinais e Sistemas para engenharia da Computação (SSC)
Módulo principal do projeto
"""

__version__ = "0.0.1"
__author__ = "Grupo 4"

from . import preprocessing
from . import feature_extraction  
from . import classification

__all__ = ['preprocessing', 'feature_extraction', 'classification']
