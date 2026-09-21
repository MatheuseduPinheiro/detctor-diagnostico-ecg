# Classificador de Imagens de ECG

Ferramenta computacional desenvolvida para **classificação de imagens de eletrocardiogramas (ECG)** utilizando técnicas de aprendizado de máquina.

O projeto classifica imagens de ECG em quatro categorias:

* **ECG normal**
* **Batimento cardíaco anormal**
* **Infarto do miocárdio**
* **Histórico de infarto do miocárdio**

A aplicação utiliza um modelo **LightGBM** selecionado a partir da comparação experimental entre diferentes algoritmos de classificação. A ferramenta foi desenvolvida como recurso computacional de apoio à análise, não constituindo um sistema de diagnóstico clínico autônomo.

## Objetivo

O objetivo do projeto é desenvolver um classificador de imagens de eletrocardiogramas capaz de diferenciar as quatro categorias consideradas e integrar o modelo selecionado a uma aplicação Web para submissão, processamento e classificação das imagens.

## Conjunto de dados

Os experimentos foram realizados utilizando o **ECG Images Dataset of Cardiac Patients**, disponibilizado no Mendeley Data.

**DOI:** `10.17632/gwbz3fsgp8.2`

O conjunto utilizado possui 929 imagens distribuídas entre quatro categorias:

| Classe                            | Imagens |
| --------------------------------- | ------: |
| Batimento cardíaco anormal        |     233 |
| Infarto do miocárdio              |     240 |
| ECG normal                        |     284 |
| Histórico de infarto do miocárdio |     172 |
| **Total**                         | **929** |

## Pré-processamento

Antes da classificação, as imagens passam pelo seguinte fluxo:

1. conversão para escala de cinza;
2. redimensionamento para `128 × 128` pixels;
3. normalização dos pixels para o intervalo entre `0` e `1`;
4. transformação da matriz da imagem em vetor unidimensional (`flatten`);
5. padronização das características utilizando `StandardScaler`.

O `StandardScaler` utilizado durante a inferência mantém os parâmetros obtidos durante o treinamento do modelo.

## Classificadores avaliados

Durante a etapa experimental foram comparados 12 algoritmos:

* Support Vector Machine (SVM);
* K-Nearest Neighbors (KNN);
* Random Forest;
* Logistic Regression;
* Decision Tree;
* Gaussian Naive Bayes;
* Gradient Boosting;
* AdaBoost;
* Extra Trees;
* XGBoost;
* LightGBM;
* CatBoost.

Os classificadores foram submetidos ao mesmo protocolo experimental para permitir a comparação dos resultados.

## Modelo selecionado

O **LightGBM** foi selecionado para integração à ferramenta a partir da análise conjunta das métricas utilizadas no experimento.

No experimento de comparação dos classificadores, foram obtidos:

| Métrica           | Resultado |
| ----------------- | --------: |
| Accuracy          |    0,8750 |
| Precision         |    0,8862 |
| Recall            |    0,8750 |
| F1-Score          |    0,8674 |
| Balanced Accuracy |    0,8473 |
| MCC               |    0,8356 |
| Cohen's Kappa     |    0,8300 |
| Mean AUC          |    0,9639 |

Após a seleção do algoritmo, 12 imagens foram reservadas para a verificação funcional da aplicação, correspondendo a três imagens de cada categoria. Essas imagens foram mantidas fora do treinamento, validação e teste do modelo LightGBM preparado especificamente para integração à ferramenta.

## Análise de incerteza

A aplicação também possui uma etapa complementar destinada à caracterização da variabilidade das probabilidades produzidas pelo classificador.

Foram utilizados modelos LightGBM obtidos por reamostragem **bootstrap**. A partir das probabilidades produzidas para a classe indicada pelo modelo principal, são calculados:

* probabilidade média;
* desvio-padrão;
* erro-padrão;
* intervalo de confiança de 95%.

Essas informações representam a variabilidade observada entre os modelos e não devem ser interpretadas como probabilidade de a classificação estar clinicamente correta.

## Arquitetura

A aplicação foi desenvolvida em **Python** utilizando **Flask**.

A organização do projeto segue uma separação entre aplicação, controlador, serviços, modelo e configuração.

```text
detctor-diagnostico-ecg/
│
├── main.py
├── config.py
│
├── Controller/
│   └── route.py
│
├── Services/
│   ├── image_service.py
│   ├── prediction_service.py
│   ├── preprocessing.py
│   ├── uncertainty.py
│   ├── metrics.py
│   └── plots.py
│
├── Model/
│   └── machine.py
│
├── Dataset_img/
├── dump/
└── ...
```

### Principais componentes

**`main.py`**
Inicializa a aplicação Flask, registra o `Blueprint` e inicia o servidor.

**`Controller/route.py`**
Gerencia as rotas HTTP, recebe as imagens submetidas pelo usuário e aciona os serviços da aplicação.

**`Services/image_service.py`**
Realiza validação da extensão, armazenamento temporário e remoção das imagens.

**`Services/preprocessing.py`**
Executa o mesmo fluxo de pré-processamento utilizado para preparar as imagens para o classificador.

**`Services/prediction_service.py`**
Orquestra o processo de classificação, incluindo carregamento do modelo, pré-processamento, inferência, métricas e organização dos resultados.

**`Services/uncertainty.py`**
Calcula as informações relacionadas à variabilidade das probabilidades utilizando os modelos bootstrap.

**`Services/metrics.py`**
Organiza e formata as métricas apresentadas pela aplicação.

**`Services/plots.py`**
Produz as representações gráficas utilizadas pela interface, incluindo matriz de confusão e curva ROC.

**`Model/machine.py`**
Contém os procedimentos relacionados ao treinamento, avaliação e serialização do modelo LightGBM.

## Fluxo da aplicação

O fluxo básico da ferramenta é:

```text
Imagem de ECG
      ↓
Validação do arquivo
      ↓
Pré-processamento
      ↓
LightGBM
      ↓
Predição das quatro classes
      ↓
Análise de incerteza
      ↓
Organização das métricas
      ↓
Apresentação dos resultados
```

O usuário seleciona uma imagem nos formatos permitidos (`PNG`, `JPG` ou `JPEG`). A imagem é armazenada temporariamente, pré-processada e submetida ao classificador.

Após a inferência, a aplicação apresenta a classe sugerida e as probabilidades associadas às quatro categorias, juntamente com informações complementares do modelo.

O arquivo temporário é removido após o processamento.

## Tecnologias

O projeto utiliza principalmente:

* Python;
* Flask;
* LightGBM;
* scikit-learn;
* NumPy;
* Matplotlib;
* HTML;
* CSS;
* JavaScript.

Outras dependências utilizadas pelo projeto devem ser consultadas nos arquivos de configuração do ambiente.

## Execução

Clone o repositório:

```bash
git clone git@github.com:MatheuseduPinheiro/detctor-diagnostico-ecg.git
```

Entre no diretório:

```bash
cd detctor-diagnostico-ecg
```

Instale as dependências do projeto conforme o ambiente configurado.

Execute a aplicação:

```bash
python3 main.py
```

Após a inicialização do Flask, acesse no navegador o endereço informado pelo servidor.

## Verificação funcional

Foram reservadas 12 imagens, sendo três de cada uma das quatro classes, para verificar o fluxo integrado da aplicação.

Essa etapa verifica o funcionamento do processo:

```text
seleção da imagem
        ↓
submissão
        ↓
pré-processamento
        ↓
classificação
        ↓
apresentação do resultado
```

Essas amostras não participaram do treinamento, validação ou teste do modelo LightGBM preparado especificamente para a aplicação.

A verificação funcional não corresponde a uma validação clínica da ferramenta.

## Limitações

O projeto deve ser interpretado dentro do escopo experimental em que foi desenvolvido.

Entre as principais limitações estão:

* utilização de um único conjunto de dados como base experimental;
* classificação restrita às quatro categorias disponíveis e consideradas no estudo;
* ausência de validação clínica da ferramenta;
* ausência de avaliação da ferramenta com profissionais de saúde;
* resultados de desempenho associados ao conjunto de dados e ao protocolo experimental utilizados.

Portanto, os resultados não devem ser generalizados automaticamente para outras populações, equipamentos, formatos de ECG ou contextos clínicos.

## Uso acadêmico

Este projeto integra uma pesquisa acadêmica relacionada à classificação computacional de imagens de eletrocardiogramas.

A ferramenta é concebida como um **recurso computacional de apoio à análise**. A classificação produzida pelo modelo não substitui a interpretação do ECG nem a decisão clínica realizada por profissional de saúde.

## Autor

**Matheus Pinheiro**

GitHub: `MatheuseduPinheiro`
