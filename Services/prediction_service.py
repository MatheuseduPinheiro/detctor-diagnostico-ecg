import joblib

from config import (
    MODEL_PATH,
    nome_amigavel
)

from Services.preprocessing import (
    processar_imagem
)

from Services.uncertainty import (
    calcular_incerteza_predicao
)

from Services.metrics import (
    preparar_metricas
)

from Services.plots import (
    gerar_grafico_matriz_confusao,
    gerar_grafico_curva_roc
)

from Services.image_service import (
    imagem_original_para_base64
)


# ============================================================
# CARREGA PKL
# ============================================================

def carregar_pacote():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}"
        )


    pacote = joblib.load(
        MODEL_PATH
    )


    campos = {

        "modelo",
        "scaler",
        "classes",
        "modelos_incerteza",
        "configuracao_incerteza"

    }


    ausentes = (
        campos
        -
        set(
            pacote.keys()
        )
    )


    if ausentes:

        raise ValueError(

            "Campos ausentes no PKL: "
            +
            ", ".join(
                ausentes
            )

        )


    return pacote


# ============================================================
# AUXILIAR DE DATASET
# ============================================================

def obter_quantidade(
    dados
):

    if isinstance(
        dados,
        dict
    ):

        return dados.get(
            "Quantidade",
            "Não informado"
        )


    return dados


def obter_percentual(
    dados
):

    if isinstance(
        dados,
        dict
    ):

        return dados.get(
            "Percentual",
            "Não informado"
        )


    return "Não informado"


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def classificar_ecg(
    caminho_temporario,
    extensao
):

    # --------------------------------------------------------
    # CARREGA PACOTE
    # --------------------------------------------------------

    pacote = carregar_pacote()


    modelo = pacote[
        "modelo"
    ]

    scaler = pacote[
        "scaler"
    ]

    classes = pacote[
        "classes"
    ]

    modelos_incerteza = pacote[
        "modelos_incerteza"
    ]

    configuracao_incerteza = pacote[
        "configuracao_incerteza"
    ]


    image_size = tuple(

        pacote.get(
            "image_size",
            (
                128,
                128
            )
        )

    )


    # --------------------------------------------------------
    # MAPEAMENTO DAS CLASSES
    # --------------------------------------------------------

    id_to_class = {

        int(id_classe):
            classe

        for classe, id_classe
        in classes.items()

    }


    # --------------------------------------------------------
    # IMAGEM PARA EXIBIÇÃO
    # --------------------------------------------------------

    imagem_enviada = (
        imagem_original_para_base64(
            caminho_temporario,
            extensao
        )
    )


    # --------------------------------------------------------
    # PRÉ-PROCESSAMENTO
    # --------------------------------------------------------

    imagem = processar_imagem(
        caminho_temporario,
        scaler,
        image_size
    )


    # --------------------------------------------------------
    # PREDIÇÃO
    # --------------------------------------------------------

    predicao = int(

        modelo.predict(
            imagem
        )[0]

    )


    probabilidades_modelo = (
        modelo.predict_proba(
            imagem
        )[0]
    )


    if predicao not in id_to_class:

        raise ValueError(
            f"Classe desconhecida: {predicao}"
        )


    classe_original = (
        id_to_class[
            predicao
        ]
    )


    classe_resultado = (
        nome_amigavel(
            classe_original
        )
    )


    # --------------------------------------------------------
    # PROBABILIDADES
    # --------------------------------------------------------

    probabilidades = {}


    for classe_id, probabilidade in zip(
        modelo.classes_,
        probabilidades_modelo
    ):

        classe_id = int(
            classe_id
        )


        if classe_id not in id_to_class:

            continue


        classe_interna = (
            id_to_class[
                classe_id
            ]
        )


        classe_exibicao = (
            nome_amigavel(
                classe_interna
            )
        )


        probabilidades[
            classe_exibicao
        ] = round(
            float(
                probabilidade
            )
            *
            100,
            2
        )


    probabilidades = dict(

        sorted(

            probabilidades.items(),

            key=lambda item:
                item[1],

            reverse=True

        )

    )


    # --------------------------------------------------------
    # CONFIANÇA DO MODELO PRINCIPAL
    # --------------------------------------------------------

    classes_modelo = [

        int(
            classe
        )

        for classe
        in modelo.classes_

    ]


    posicao = (
        classes_modelo.index(
            predicao
        )
    )


    confianca = round(

        float(
            probabilidades_modelo[
                posicao
            ]
        )
        *
        100,

        2

    )


    # --------------------------------------------------------
    # INCERTEZA
    # --------------------------------------------------------

    incerteza_predicao = (
        calcular_incerteza_predicao(

            modelos_incerteza=
                modelos_incerteza,

            imagem=
                imagem,

            classe_predita=
                predicao,

            configuracao_incerteza=
                configuracao_incerteza

        )
    )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = pacote.get(
        "dataset",
        {}
    )


    divisao = dataset.get(
        "Divisao dos Dados",
        {}
    )


    total_amostras = dataset.get(
        "Total de Amostras",
        "Não informado"
    )


    treino_dados = divisao.get(
        "Treino",
        {}
    )


    validacao_dados = divisao.get(

        "Validacao",

        divisao.get(
            "Validação",
            {}
        )

    )


    teste_dados = divisao.get(
        "Teste",
        {}
    )


    # --------------------------------------------------------
    # RESULTADOS DO TESTE
    # --------------------------------------------------------

    resultado_teste = pacote.get(
        "resultado_teste",
        {}
    )


    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    metricas = preparar_metricas(
        resultado_teste,
        classe_original
    )


    # --------------------------------------------------------
    # MATRIZ DE CONFUSÃO
    # --------------------------------------------------------

    matriz_confusao = resultado_teste.get(

        "Matriz de Confusão",

        resultado_teste.get(
            "Matriz de Confusao"
        )

    )


    # --------------------------------------------------------
    # CURVA ROC
    # --------------------------------------------------------

    curvas_roc = resultado_teste.get(

        "Curvas ROC",

        resultado_teste.get(
            "ROC"
        )

    )


    classes_grafico = [

        nome_amigavel(
            id_to_class[
                classe_id
            ]
        )

        for classe_id
        in sorted(
            id_to_class.keys()
        )

    ]


    grafico_matriz_confusao = (
        gerar_grafico_matriz_confusao(
            matriz_confusao,
            classes_grafico
        )
    )


    grafico_curva_roc = (
        gerar_grafico_curva_roc(
            curvas_roc
        )
    )


    # --------------------------------------------------------
    # RETORNO PARA O CONTROLLER
    # --------------------------------------------------------

    return {

        "imagem_enviada":
            imagem_enviada,

        "classe":
            classe_resultado,

        "classe_original":
            classe_original,

        "confianca":
            confianca,

        "probabilidades":
            probabilidades,

        "incerteza_predicao":
            incerteza_predicao,


        # MÉTRICAS

        **metricas,


        # GRÁFICOS

        "grafico_matriz_confusao":
            grafico_matriz_confusao,

        "grafico_curva_roc":
            grafico_curva_roc,


        # DATASET

        "total_amostras":
            total_amostras,

        "treino_quantidade":
            obter_quantidade(
                treino_dados
            ),

        "treino_percentual":
            obter_percentual(
                treino_dados
            ),

        "validacao_quantidade":
            obter_quantidade(
                validacao_dados
            ),

        "validacao_percentual":
            obter_percentual(
                validacao_dados
            ),

        "teste_quantidade":
            obter_quantidade(
                teste_dados
            ),

        "teste_percentual":
            obter_percentual(
                teste_dados
            ),


        # PRÉ-PROCESSAMENTO

        "image_width":
            image_size[0],

        "image_height":
            image_size[1],

        "grayscale":
            pacote.get(
                "grayscale",
                True
            ),

        "normalizacao_pixels":
            "0 a 1",

        "flatten":
            pacote.get(
                "flatten",
                True
            ),

        "algoritmo":
            "LightGBM"

    }