import base64
import uuid

from pathlib import Path

from config import (
    EXTENSOES_PERMITIDAS,
    UPLOAD_DIR
)


# ============================================================
# EXTENSÃO PERMITIDA
# ============================================================

def extensao_permitida(
    filename
):

    return (
        Path(
            filename
        ).suffix.lower()
        in EXTENSOES_PERMITIDAS
    )


# ============================================================
# SALVA IMAGEM TEMPORÁRIA
# ============================================================

def salvar_imagem_temporaria(
    arquivo
):

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


    return (
        caminho_temporario,
        extensao
    )


# ============================================================
# IMAGEM ORIGINAL -> BASE64
# ============================================================

def imagem_original_para_base64(
    caminho,
    extensao
):

    with open(
        caminho,
        "rb"
    ) as arquivo_imagem:

        dados = base64.b64encode(
            arquivo_imagem.read()
        ).decode(
            "utf-8"
        )


    if extensao == ".png":

        mime_type = "image/png"

    else:

        mime_type = "image/jpeg"


    return (
        f"data:{mime_type};base64,"
        +
        dados
    )


# ============================================================
# REMOVE ARQUIVO TEMPORÁRIO
# ============================================================

def remover_imagem_temporaria(
    caminho
):

    if (
        caminho is None
        or not caminho.exists()
    ):

        return


    try:

        caminho.unlink()


    except Exception as erro_remocao:

        print(
            "Erro ao remover arquivo temporário:",
            erro_remocao
        )