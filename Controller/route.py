from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for
)

from pathlib import Path

import base64
import io
import uuid

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# BLUEPRINT
# ============================================================

app_blueprint = Blueprint(
    "app",
    __name__
)


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


UPLOAD_DIR = (
    BASE_DIR /
    "uploads"
)


MODEL_PATH = (
    BASE_DIR /
    "dump" /
    "LightGBM-model.pkl"
)


UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXTENSÕES
# ============================================================

EXTENSOES_PERMITIDAS = {
    ".png",
    ".jpg",
    ".jpeg"
}


# ============================================================
# NOMES AMIGÁVEIS
# ============================================================

NOMES_CLASSES = {

    "abnormal_heartbeat_ecg_images":
        "Ritmo Anormal",

    "myocardial_infarction_ecg_images":
        "Infarto do Miocárdio",

    "normal_ecg_images":
        "ECG Normal",

    "post_mi_history_ecg_images":
        "Histórico de Infarto"

}


def nome_amigavel(classe):

    return NOMES_CLASSES.get(
        classe,
        classe
    )


# ============================================================
# EXTENSÃO PERMITIDA
# ============================================================

def extensao_permitida(filename):

    return (
        Path(filename).suffix.lower()
        in EXTENSOES_PERMITIDAS
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
        "classes"
    }


    ausentes = (
        campos -
        set(pacote.keys())
    )


    if ausentes:

        raise ValueError(
            "Campos ausentes no PKL: "
            +
            ", ".join(ausentes)
        )


    return pacote


# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def processar_imagem(
    caminho,
    scaler,
    image_size
):

    imagem = cv2.imread(
        str(caminho),
        cv2.IMREAD_GRAYSCALE
    )


    if imagem is None:

        raise ValueError(
            "Não foi possível ler a imagem enviada."
        )


    imagem = cv2.resize(
        imagem,
        tuple(image_size)
    )


    imagem = (
        imagem /
        255.0
    )


    imagem = imagem.flatten()


    imagem = imagem.reshape(
        1,
        -1
    )


    imagem = scaler.transform(
        imagem
    )


    return imagem


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
# FORMATA AUC / MCC / KAPPA
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
        range(quantidade)
    )


    eixo.set_yticks(
        range(quantidade)
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


# ============================================================
# ROTA /
# ============================================================

@app_blueprint.route("/")
def root():

    return redirect(
        url_for(
            "app.home"
        )
    )


# ============================================================
# PÁGINA PRINCIPAL
# ============================================================

@app_blueprint.route(
    "/classificacao-ecg",
    methods=[
        "GET"
    ]
)
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# RESULTADOS
# ============================================================

@app_blueprint.route(
    "/classificacao-ecg/resultados",
    methods=[
        "GET",
        "POST"
    ]
)
def predict():


    # --------------------------------------------------------
    # ACESSO DIRETO PELO NAVEGADOR
    # --------------------------------------------------------

    if request.method == "GET":

        return redirect(
            url_for(
                "app.home"
            )
        )


    caminho_temporario = None


    try:

        # ====================================================
        # UPLOAD
        # ====================================================

        if "imagem" not in request.files:

            return render_template(
                "index.html",
                erro="Nenhuma imagem foi enviada."
            ), 400


        arquivo = request.files[
            "imagem"
        ]


        if arquivo.filename == "":

            return render_template(
                "index.html",
                erro="Nenhuma imagem foi selecionada."
            ), 400


        if not extensao_permitida(
            arquivo.filename
        ):

            return render_template(
                "index.html",
                erro=(
                    "Formato não permitido. "
                    "Utilize PNG, JPG ou JPEG."
                )
            ), 400


        # ====================================================
        # MODELO
        # ====================================================

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


        image_size = tuple(
            pacote.get(
                "image_size",
                (
                    128,
                    128
                )
            )
        )


        id_to_class = {

            int(id_classe):
                classe

            for classe, id_classe
            in classes.items()

        }


        # ====================================================
        # SALVA IMAGEM TEMPORÁRIA
        # ====================================================

        extensao = Path(
            arquivo.filename
        ).suffix.lower()


        caminho_temporario = (
            UPLOAD_DIR /
            (
                str(
                    uuid.uuid4()
                )
                +
                extensao
            )
        )


        arquivo.save(
            caminho_temporario
        )


        # ====================================================
        # PROCESSAMENTO
        # ====================================================

        imagem = processar_imagem(
            caminho_temporario,
            scaler,
            image_size
        )


        # ====================================================
        # PREDIÇÃO
        # ====================================================

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


        # ====================================================
        # PROBABILIDADES
        # ====================================================

        probabilidades = {}


        for (
            classe_id,
            probabilidade
        ) in zip(
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


        # ====================================================
        # CONFIANÇA
        # ====================================================

        classes_modelo = [
            int(classe)
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


        # ====================================================
        # DATASET
        # ====================================================

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


        treino_quantidade = (
            treino_dados.get(
                "Quantidade",
                "Não informado"
            )
            if isinstance(
                treino_dados,
                dict
            )
            else treino_dados
        )


        treino_percentual = (
            treino_dados.get(
                "Percentual",
                "Não informado"
            )
            if isinstance(
                treino_dados,
                dict
            )
            else "Não informado"
        )


        validacao_quantidade = (
            validacao_dados.get(
                "Quantidade",
                "Não informado"
            )
            if isinstance(
                validacao_dados,
                dict
            )
            else validacao_dados
        )


        validacao_percentual = (
            validacao_dados.get(
                "Percentual",
                "Não informado"
            )
            if isinstance(
                validacao_dados,
                dict
            )
            else "Não informado"
        )


        teste_quantidade = (
            teste_dados.get(
                "Quantidade",
                "Não informado"
            )
            if isinstance(
                teste_dados,
                dict
            )
            else teste_dados
        )


        teste_percentual = (
            teste_dados.get(
                "Percentual",
                "Não informado"
            )
            if isinstance(
                teste_dados,
                dict
            )
            else "Não informado"
        )


        # ====================================================
        # RESULTADO DO TESTE
        # ====================================================

        resultado_teste = pacote.get(
            "resultado_teste",
            {}
        )


        # ====================================================
        # MÉTRICAS GLOBAIS
        # ====================================================

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


        balanced_accuracy = (
            formatar_percentual(
                resultado_teste.get(
                    "Balanced Accuracy"
                )
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


        # ====================================================
        # MÉTRICAS POR CLASSE
        # ====================================================

        metricas_brutas = resultado_teste.get(
            "Por Classe",
            {}
        )


        auc_bruto = resultado_teste.get(
            "AUC por Classe",
            {}
        )


        metricas_por_classe = {}


        for (
            classe_interna,
            metricas
        ) in metricas_brutas.items():

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


        # ====================================================
        # GRÁFICOS
        # ====================================================

        matriz_confusao = resultado_teste.get(
            "Matriz de Confusão",
            resultado_teste.get(
                "Matriz de Confusao"
            )
        )


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


        # ====================================================
        # RESULTADOS.HTML
        # ====================================================

        return render_template(

            "resultados.html",

            classe=
                classe_resultado,

            classe_original=
                classe_original,

            confianca=
                confianca,

            probabilidades=
                probabilidades,

            accuracy=
                accuracy,

            precision=
                precision,

            recall=
                recall,

            f1_score=
                f1_score,

            balanced_accuracy=
                balanced_accuracy,

            mcc=
                mcc,

            cohen_kappa=
                cohen_kappa,

            mean_auc=
                mean_auc,

            metricas_por_classe=
                metricas_por_classe,

            grafico_matriz_confusao=
                grafico_matriz_confusao,

            grafico_curva_roc=
                grafico_curva_roc,

            total_amostras=
                total_amostras,

            treino_quantidade=
                treino_quantidade,

            treino_percentual=
                treino_percentual,

            validacao_quantidade=
                validacao_quantidade,

            validacao_percentual=
                validacao_percentual,

            teste_quantidade=
                teste_quantidade,

            teste_percentual=
                teste_percentual,

            image_width=
                image_size[0],

            image_height=
                image_size[1],

            grayscale=
                pacote.get(
                    "grayscale",
                    True
                ),

            normalizacao_pixels=
                "0 a 1",

            flatten=
                pacote.get(
                    "flatten",
                    True
                ),

            algoritmo=
                "LightGBM"

        )


    except Exception as erro:

        print(
            "\nERRO DURANTE A PREDIÇÃO:"
        )

        print(
            type(erro).__name__,
            erro
        )


        return render_template(
            "index.html",
            erro=(
                f"{type(erro).__name__}: {erro}"
            )
        ), 500


    finally:

        if (
            caminho_temporario is not None
            and caminho_temporario.exists()
        ):

            try:

                caminho_temporario.unlink()

            except Exception as erro_remocao:

                print(
                    "Erro ao remover arquivo temporário:",
                    erro_remocao
                )