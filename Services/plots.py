import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import nome_amigavel


# ============================================================
# FIGURA -> BASE64
# ============================================================

def figura_para_base64(figura):

    buffer = io.BytesIO()

    figura.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=130
    )

    plt.close(
        figura
    )

    buffer.seek(0)

    dados = base64.b64encode(
        buffer.getvalue()
    ).decode(
        "utf-8"
    )

    return (
        "data:image/png;base64,"
        +
        dados
    )


# ============================================================
# MATRIZ DE CONFUSÃO
# ============================================================

def gerar_grafico_matriz_confusao(
    matriz,
    nomes_classes
):

    if matriz is None:

        return None


    try:

        matriz = matriz.tolist()

    except AttributeError:

        pass


    if not matriz:

        return None


    figura, eixo = plt.subplots(
        figsize=(
            8,
            6
        )
    )


    imagem = eixo.imshow(
        matriz
    )


    eixo.set_title(
        "Matriz de Confusão"
    )

    eixo.set_xlabel(
        "Classe predita"
    )

    eixo.set_ylabel(
        "Classe real"
    )


    quantidade = len(
        matriz
    )


    nomes_classes = (
        nomes_classes[:quantidade]
    )


    eixo.set_xticks(
        range(
            quantidade
        )
    )

    eixo.set_yticks(
        range(
            quantidade
        )
    )


    eixo.set_xticklabels(
        nomes_classes,
        rotation=30,
        ha="right"
    )

    eixo.set_yticklabels(
        nomes_classes
    )


    for linha in range(
        quantidade
    ):

        for coluna in range(
            quantidade
        ):

            eixo.text(
                coluna,
                linha,
                str(
                    matriz[
                        linha
                    ][
                        coluna
                    ]
                ),
                ha="center",
                va="center"
            )


    figura.colorbar(
        imagem,
        ax=eixo
    )

    figura.tight_layout()


    return figura_para_base64(
        figura
    )


# ============================================================
# AUXILIAR ROC
# ============================================================

def obter_campo(
    dados,
    nomes
):

    if not isinstance(
        dados,
        dict
    ):

        return None


    for nome in nomes:

        if nome in dados:

            return dados[
                nome
            ]


    return None


# ============================================================
# CURVA ROC
# ============================================================

def gerar_grafico_curva_roc(
    curvas_roc
):

    if not isinstance(
        curvas_roc,
        dict
    ):

        return None


    figura, eixo = plt.subplots(
        figsize=(
            8,
            6
        )
    )


    encontrou = False


    for classe, dados in curvas_roc.items():

        if not isinstance(
            dados,
            dict
        ):

            continue


        fpr = obter_campo(
            dados,
            [
                "fpr",
                "FPR"
            ]
        )

        tpr = obter_campo(
            dados,
            [
                "tpr",
                "TPR"
            ]
        )

        auc = obter_campo(
            dados,
            [
                "auc",
                "AUC"
            ]
        )


        if (
            fpr is None
            or tpr is None
        ):

            continue


        classe_exibicao = nome_amigavel(
            classe
        )


        if auc is not None:

            legenda = (
                f"{classe_exibicao} "
                f"(AUC = {float(auc):.3f})"
            )

        else:

            legenda = classe_exibicao


        eixo.plot(
            fpr,
            tpr,
            label=legenda
        )


        encontrou = True


    if not encontrou:

        plt.close(
            figura
        )

        return None


    eixo.plot(
        [
            0,
            1
        ],
        [
            0,
            1
        ],
        linestyle="--",
        label="Referência"
    )


    eixo.set_title(
        "Curva ROC"
    )

    eixo.set_xlabel(
        "Taxa de falsos positivos"
    )

    eixo.set_ylabel(
        "Taxa de verdadeiros positivos"
    )


    eixo.set_xlim(
        0,
        1
    )

    eixo.set_ylim(
        0,
        1.02
    )


    eixo.legend(
        loc="lower right"
    )

    eixo.grid(
        alpha=0.25
    )


    figura.tight_layout()


    return figura_para_base64(
        figura
    )