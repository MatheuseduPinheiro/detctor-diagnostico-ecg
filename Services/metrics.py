from config import nome_amigavel


# ============================================================
# FORMATA PORCENTAGEM
# ============================================================

def formatar_percentual(valor):

    if valor is None:

        return "Não informado"

    try:

        return (
            f"{float(valor) * 100:.2f}%"
            .replace(
                ".",
                ","
            )
        )

    except Exception:

        return str(valor)


# ============================================================
# FORMATA DECIMAL
# ============================================================

def formatar_decimal(
    valor,
    casas=3
):

    if valor is None:

        return "Não informado"

    try:

        return (
            f"{float(valor):.{casas}f}"
            .replace(
                ".",
                ","
            )
        )

    except Exception:

        return str(valor)


# ============================================================
# PREPARA MÉTRICAS
# ============================================================

def preparar_metricas(
    resultado_teste,
    classe_original
):

    # --------------------------------------------------------
    # MÉTRICAS GLOBAIS
    # --------------------------------------------------------

    accuracy = formatar_percentual(
        resultado_teste.get(
            "Accuracy"
        )
    )

    precision = formatar_percentual(
        resultado_teste.get(
            "Precision"
        )
    )

    recall = formatar_percentual(
        resultado_teste.get(
            "Recall"
        )
    )

    f1_score = formatar_percentual(
        resultado_teste.get(
            "F1-Score"
        )
    )

    balanced_accuracy = formatar_percentual(
        resultado_teste.get(
            "Balanced Accuracy"
        )
    )

    mcc = formatar_decimal(
        resultado_teste.get(
            "MCC"
        )
    )

    cohen_kappa = formatar_decimal(
        resultado_teste.get(
            "Cohen's Kappa"
        )
    )

    mean_auc = formatar_decimal(
        resultado_teste.get(
            "Mean AUC"
        )
    )


    # --------------------------------------------------------
    # MÉTRICAS POR CLASSE
    # --------------------------------------------------------

    metricas_brutas = resultado_teste.get(
        "Por Classe",
        {}
    )

    auc_bruto = resultado_teste.get(
        "AUC por Classe",
        {}
    )

    metricas_por_classe = {}


    for classe_interna, metricas in metricas_brutas.items():

        classe_exibicao = nome_amigavel(
            classe_interna
        )

        auc_valor = auc_bruto.get(
            classe_interna
        )


        metricas_por_classe[
            classe_exibicao
        ] = {

            "sensibilidade":
                formatar_percentual(
                    metricas.get(
                        "Sensibilidade"
                    )
                ),

            "especificidade":
                formatar_percentual(
                    metricas.get(
                        "Especificidade"
                    )
                ),

            "npv":
                formatar_percentual(
                    metricas.get(
                        "NPV"
                    )
                ),

            "auc":
                formatar_decimal(
                    auc_valor
                )

        }


    # --------------------------------------------------------
    # MATRIZ DE CONFUSÃO DA CLASSE ESCOLHIDA
    # --------------------------------------------------------

    dados_classe_escolhida = (
        metricas_brutas.get(
            classe_original,
            {}
        )
    )


    interpretacao_matriz = {

        "verdadeiros_positivos":
            int(
                dados_classe_escolhida.get(
                    "TP",
                    0
                )
            ),

        "falsos_negativos":
            int(
                dados_classe_escolhida.get(
                    "FN",
                    0
                )
            ),

        "falsos_positivos":
            int(
                dados_classe_escolhida.get(
                    "FP",
                    0
                )
            ),

        "verdadeiros_negativos":
            int(
                dados_classe_escolhida.get(
                    "TN",
                    0
                )
            )

    }


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1_score":
            f1_score,

        "balanced_accuracy":
            balanced_accuracy,

        "mcc":
            mcc,

        "cohen_kappa":
            cohen_kappa,

        "mean_auc":
            mean_auc,

        "metricas_por_classe":
            metricas_por_classe,

        "interpretacao_matriz":
            interpretacao_matriz

    }