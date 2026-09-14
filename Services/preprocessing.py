import cv2


# ============================================================
# PRÉ-PROCESSAMENTO DA IMAGEM
# ============================================================

def processar_imagem(
    caminho,
    scaler,
    image_size
):

    # --------------------------------------------------------
    # LEITURA EM ESCALA DE CINZA
    # --------------------------------------------------------

    imagem = cv2.imread(
        str(caminho),
        cv2.IMREAD_GRAYSCALE
    )

    if imagem is None:

        raise ValueError(
            "Não foi possível ler a imagem enviada."
        )


    # --------------------------------------------------------
    # REDIMENSIONAMENTO
    # --------------------------------------------------------

    imagem = cv2.resize(
        imagem,
        tuple(image_size)
    )


    # --------------------------------------------------------
    # NORMALIZAÇÃO DOS PIXELS
    # --------------------------------------------------------

    imagem = (
        imagem /
        255.0
    )


    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    imagem = imagem.flatten()

    imagem = imagem.reshape(
        1,
        -1
    )


    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

    imagem = scaler.transform(
        imagem
    )


    return imagem