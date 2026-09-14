from pathlib import Path


# ============================================================
# CAMINHOS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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
# EXTENSÕES PERMITIDAS
# ============================================================

EXTENSOES_PERMITIDAS = {
    ".png",
    ".jpg",
    ".jpeg"
}


# ============================================================
# NOMES DAS CLASSES
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