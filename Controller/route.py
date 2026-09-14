from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for
)

from Service.image_service import (
    extensao_permitida,
    salvar_imagem_temporaria,
    remover_imagem_temporaria
)

from Service.prediction_service import (
    classificar_ecg
)


# ============================================================
# BLUEPRINT
# ============================================================

app_blueprint = Blueprint(
    "app",
    __name__
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

        # ----------------------------------------------------
        # VERIFICA SE O ARQUIVO FOI ENVIADO
        # ----------------------------------------------------

        if "imagem" not in request.files:

            return render_template(
                "index.html",
                erro="Nenhuma imagem foi enviada."
            ), 400


        arquivo = request.files[
            "imagem"
        ]


        # ----------------------------------------------------
        # VERIFICA SE UMA IMAGEM FOI SELECIONADA
        # ----------------------------------------------------

        if arquivo.filename == "":

            return render_template(
                "index.html",
                erro="Nenhuma imagem foi selecionada."
            ), 400


        # ----------------------------------------------------
        # VALIDA EXTENSÃO
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SALVA TEMPORARIAMENTE
        # ----------------------------------------------------

        caminho_temporario, extensao = (
            salvar_imagem_temporaria(
                arquivo
            )
        )


        # ----------------------------------------------------
        # CHAMA O SERVIÇO DE CLASSIFICAÇÃO
        # ----------------------------------------------------

        resultado = classificar_ecg(
            caminho_temporario,
            extensao
        )


        # ----------------------------------------------------
        # ENVIA RESULTADOS PARA O HTML
        # ----------------------------------------------------

        return render_template(
            "resultados.html",
            **resultado
        )


    # --------------------------------------------------------
    # ERROS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # REMOVE ARQUIVO TEMPORÁRIO
    # --------------------------------------------------------

    finally:

        remover_imagem_temporaria(
            caminho_temporario
        )