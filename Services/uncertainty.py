import numpy as np


# ============================================================
# INCERTEZA DA PREDIÇÃO
# ============================================================

def calcular_incerteza_predicao(
    modelos_incerteza,
    imagem,
    classe_predita,
    configuracao_incerteza
):

    if not modelos_incerteza:

        raise ValueError(
            "Nenhum modelo de incerteza foi encontrado "
            "no arquivo PKL."
        )


    probabilidades_classe = []


    # --------------------------------------------------------
    # EXECUTA OS MODELOS BOOTSTRAP
    # --------------------------------------------------------

    for numero_modelo, modelo_incerteza in enumerate(
        modelos_incerteza,
        start=1
    ):

        classes_modelo_incerteza = [

            int(classe)

            for classe
            in modelo_incerteza.classes_

        ]


        if classe_predita not in classes_modelo_incerteza:

            raise ValueError(

                "A classe prevista pelo modelo principal "
                f"({classe_predita}) não foi encontrada no modelo "
                f"de incerteza número {numero_modelo}."

            )


        posicao_classe = (
            classes_modelo_incerteza.index(
                classe_predita
            )
        )


        probabilidades_modelo = (
            modelo_incerteza.predict_proba(
                imagem
            )[0]
        )


        probabilidades_classe.append(

            float(
                probabilidades_modelo[
                    posicao_classe
                ]
            )

        )


    # --------------------------------------------------------
    # CONVERTE PARA ARRAY
    # --------------------------------------------------------

    probabilidades_classe = np.array(
        probabilidades_classe,
        dtype=float
    )


    numero_execucoes = int(
        len(
            probabilidades_classe
        )
    )


    # --------------------------------------------------------
    # PROBABILIDADE MÉDIA
    # --------------------------------------------------------

    probabilidade_media = float(
        np.mean(
            probabilidades_classe
        )
    )


    # --------------------------------------------------------
    # DESVIO-PADRÃO
    # --------------------------------------------------------

    if numero_execucoes > 1:

        desvio_padrao = float(

            np.std(
                probabilidades_classe,
                ddof=1
            )

        )

    else:

        desvio_padrao = 0.0


    # --------------------------------------------------------
    # ERRO-PADRÃO
    # --------------------------------------------------------

    erro_padrao = (
        desvio_padrao
        /
        np.sqrt(
            numero_execucoes
        )
    )


    # --------------------------------------------------------
    # CONFIGURAÇÃO DO INTERVALO
    # --------------------------------------------------------

    nivel_confianca = float(

        configuracao_incerteza.get(
            "nivel_confianca",
            0.95
        )

    )


    z_score = float(

        configuracao_incerteza.get(
            "z_score",
            1.96
        )

    )


    # --------------------------------------------------------
    # INTERVALO DE CONFIANÇA
    # --------------------------------------------------------

    intervalo_inferior = float(

        np.clip(

            probabilidade_media
            -
            z_score * erro_padrao,

            0.0,
            1.0

        )

    )


    intervalo_superior = float(

        np.clip(

            probabilidade_media
            +
            z_score * erro_padrao,

            0.0,
            1.0

        )

    )


    metodo = configuracao_incerteza.get(
        "metodo",
        "Bootstrap Ensemble"
    )


    # --------------------------------------------------------
    # RETORNO
    # --------------------------------------------------------

    return {

        "probabilidade_media":
            round(
                probabilidade_media * 100,
                2
            ),

        "desvio_padrao":
            round(
                desvio_padrao * 100,
                2
            ),

        "intervalo_inferior":
            round(
                intervalo_inferior * 100,
                2
            ),

        "intervalo_superior":
            round(
                intervalo_superior * 100,
                2
            ),

        "numero_execucoes":
            numero_execucoes,

        "nivel_confianca":
            round(
                nivel_confianca * 100,
                0
            ),

        "metodo":
            metodo,

        "explicacao_probabilidade_media": (
            "A mesma imagem de ECG é avaliada pelos modelos "
            "gerados nas reamostragens bootstrap. Cada modelo "
            "fornece uma probabilidade para a classe selecionada "
            "pelo modelo principal. Este valor corresponde à média "
            "dessas probabilidades."
        ),

        "explicacao_desvio_padrao": (
            "Mostra quanto as probabilidades produzidas pelos modelos "
            "variaram em torno da média. Um valor menor indica que os "
            "modelos produziram probabilidades mais semelhantes; um "
            "valor maior indica maior variação entre as execuções."
        ),

        "explicacao_intervalo_confianca": (
            "Expressa a incerteza em torno da probabilidade média "
            "estimada. O intervalo não significa que exista essa "
            "porcentagem de chance de a classificação estar correta."
        ),

        "explicacao_numero_execucoes": (
            "Indica quantos modelos LightGBM treinados com "
            "reamostragens bootstrap foram utilizados para calcular "
            "a média, o desvio-padrão e o intervalo de confiança."
        )

    }