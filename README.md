# Classificador de Doenças Cardíacas com ECG - ES413

Este projeto desenvolve um sistema de classificação automática de doenças cardíacas utilizando sinais de eletrocardiograma (ECG), aplicando técnicas avançadas de processamento de sinais digitais e machine learning. Desenvolvido para a disciplina ES413 - Sinais e Sistemas para Engenharia da Computação.

## Objetivo

Implementar um pipeline completo para:
- Pré-processamento de sinais ECG do dataset PTB-XL
- Extração de características relevantes no domínio do tempo e frequência
- Classificação automática de condições cardíacas
- Avaliação e validação dos modelos desenvolvidos

## Dataset

**PTB-XL**: Maior dataset público de ECG disponível
- **21.837 registros** de ECG de 10 segundos
- **12 derivações** padrão (I, II, III, aVR, aVL, aVF, V1-V6)
- **Frequência de amostragem**: 100 Hz e 500 Hz
- **Classes principais**: NORM, MI, STTC, CD, HYP

## Estrutura do Projeto

```
Projeto_SSC/
├── README.md                           # Documentação principal
├── requirements.txt                    # Dependências Python
├── .gitignore                         # Arquivos ignorados pelo Git
├── notebook/                          # Notebooks Jupyter (pipeline principal)
│   ├── I_preprocessamento_sinais.ipynb    # Pré-processamento de sinais ECG
│   ├── II_extracao_caracteristicas.ipynb # Extração de features
│   ├── III_classificacao_algoritmos.ipynb # Treinamento de modelos
│   └── IV_avaliacao_resultados.ipynb     # Análise e avaliação
├── src/                               # Código fonte modularizado
│   ├── preprocessing/                     # Módulos de pré-processamento
│   ├── feature_extraction/               # Extratores de características
│   └── classification/                   # Algoritmos de classificação
├── data/                              # Dados do projeto
│   ├── raw/                              # Dados brutos (PTB-XL)
│   └── features/                         # Características extraídas
├── docs/                              # Documentação técnica
│   ├── referencias/                      # Bibliografia e referências
│   └── *.pdf                            # Relatórios técnicos
└── results/                           # Resultados e modelos
    └── classification/                   # Modelos treinados e métricas
```

## Tecnologias Utilizadas

### Bibliotecas Principais
- **NumPy & SciPy**: Computação científica e processamento de sinais
- **Pandas**: Manipulação e análise de dados
- **WFDB**: Leitura de dados biomédicos PTB-XL
- **Scikit-learn**: Machine learning e avaliação de modelos
- **Matplotlib & Seaborn**: Visualização de dados

### Algoritmos Implementados
- **Pré-processamento**: Filtragem digital, normalização, remoção de baseline
- **Extração de features**: Características temporais, espectrais e entropia
- **Classificação**: Random Forest com validação cruzada
- **Avaliação**: Métricas clínicas e análise de confusão

## Como Executar

### 1. Preparação do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/SapoSopa/Projeto_SSC.git
cd Projeto_SSC

# Instalar dependências
pip install -r requirements.txt

# Baixar dataset PTB-XL (instruções no Notebook I)
```

### 2. Pipeline de Execução

Execute os notebooks na seguinte ordem:

```bash
# 1. Pré-processamento dos sinais ECG
jupyter notebook notebook/I_preprocessamento_sinais.ipynb

# 2. Extração de características
jupyter notebook notebook/II_extracao_caracteristicas.ipynb

# 3. Treinamento dos classificadores
jupyter notebook notebook/III_classificacao_algoritmos.ipynb

# 4. Avaliação e análise dos resultados
jupyter notebook notebook/IV_avaliacao_resultados.ipynb
```

### 3. Estrutura de Dados Esperada

```
data/raw/
└── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
    ├── ptbxl_database.csv
    ├── records100/
    └── records500/
```

## Documentação Técnica

- [`/docs/referencias/bibliografia.md`](docs/referencias/bibliografia.md) - Referências bibliográficas
- [`/src/preprocessing/preprocessamento_tecnica.md`](src/preprocessing/preprocessamento_tecnica.md) - Documentação do módulo de pré-processamento
- [`/src/feature_extraction/extract_features_tecnica.md`](src/feature_extraction/extract_features_tecnica.md) - Documentação da extração de características
- [`/docs/*.pdf`](docs/) - Relatórios técnicos detalhados

## Metodologia Científica

### Pipeline de Processamento
1. **Carregamento**: Dados PTB-XL em formato WFDB
2. **Pré-processamento**: Filtro passa-banda (0.5-45 Hz), normalização Z-score
3. **Extração de Features**: 25 características temporais e espectrais por canal
4. **Classificação**: Random Forest com GridSearch e validação cruzada
5. **Avaliação**: Métricas clínicas e interpretação médica

### Validação
- **Divisão estratificada**: 80% treino / 20% teste
- **Validação cruzada**: 5-fold para otimização de hiperparâmetros
- **Métricas clínicas**: Focadas em sensibilidade e especificidade

## Limitações Atuais
- Performance moderada (79.25% acurácia)
- Necessita validação clínica externa
- Desbalanceamento entre classes

## Requisitos do Sistema

- **Python**: 3.8 ou superior
- **Espaço em disco**: ~5GB para dataset PTB-XL
- **Sistema operacional**: Windows, Linux ou macOS

## Colaboradores

| [<img src="https://avatars.githubusercontent.com/u/161069298?v=4" width=115><br><sub>Esdras Gabriel</sub>](https://github.com/BlackSardes) | [<img src="https://avatars.githubusercontent.com/u/162517004?v=4" width=115><br><sub>Gabriel Campos</sub>](https://github.com/gjcms) | [<img src="https://avatars.githubusercontent.com/u/129231720?v=4" width=115><br><sub>Henrique César</sub>](https://github.com/SapoSopa) | [<img src="https://avatars.githubusercontent.com/u/98539736?v=4" width=115><br><sub>João Victor</sub>](https://github.com/jambis-prg) | [<img src="https://avatars.githubusercontent.com/u/96800329?v=4" width=115><br><sub>Luiz Gustavo</sub>](https://github.com/Zed201) | [<img src="https://avatars.githubusercontent.com/u/123107373?v=4" width=115><br><sub>Márcio Júnior</sub>](https://github.com/MAACJR032) |
| :---: | :---: | :---: | :---: | :---: | :---: |

## Licença

Este projeto foi desenvolvido para fins acadêmicos na disciplina ES413 - Sinais e Sistemas para Engenharia da Computação.

---