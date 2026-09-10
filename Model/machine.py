# ============================================================
# machine.py
# Treinamento, avaliação e serialização do modelo LightGBM
# ============================================================

import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score
)

from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA_ROOT = os.path.join(
    os.getcwd(),
    "Dataset_img"
)


# ============================================================
# PASTA DO MODELO
# ============================================================

DUMP_DIR = os.path.join(
    os.getcwd(),
    "dump"
)


# Cria a pasta dump caso ela ainda não exista
os.makedirs(
    DUMP_DIR,
    exist_ok=True
)


# Arquivo final do modelo
MODEL_OUTPUT = os.path.join(
    DUMP_DIR,
    "LightGBM-model.pkl"
)


IMAGE_SIZE = (128, 128)
RANDOM_STATE = 42


# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def load_dataset(root):

    if not os.path.exists(root):

        raise FileNotFoundError(
            f"Dataset folder not found:\n{root}"
        )

    paths = []
    labels = []
    label_map = {}
    idx = 0

    for class_name in sorted(os.listdir(root)):

        class_dir = os.path.join(
            root,
            class_name
        )

        if not os.path.isdir(class_dir):
            continue

        label_map[class_name] = idx

        for fname in os.listdir(class_dir):

            if fname.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ):

                img_path = os.path.join(
                    class_dir,
                    fname
                )

                paths.append(img_path)
                labels.append(idx)

        idx += 1

    print("\nLabel map:")
    print(label_map)

    return (
        np.array(paths),
        np.array(labels),
        label_map
    )


# ============================================================
# PRÉ-PROCESSAMENTO DE UMA IMAGEM
# ============================================================

def process_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:

        raise ValueError(
            f"Não foi possível ler a imagem:\n{path}"
        )

    img = cv2.resize(
        img,
        IMAGE_SIZE
    )

    img = img / 255.0

    img = img.flatten()

    return img


# ============================================================
# AVALIAÇÃO DO MODELO
# ============================================================

def evaluate_model(
    model_name,
    model,
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
    CLASSES
):

    # ========================================================
    # TREINAMENTO
    # ========================================================

    model.fit(
        X_train_processed,
        y_train
    )

    # ========================================================
    # PREDIÇÕES
    # ========================================================

    y_pred = model.predict(
        X_test_processed
    )

    y_prob = model.predict_proba(
        X_test_processed
    )

    # ========================================================
    # MÉTRICAS GLOBAIS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    kappa = cohen_kappa_score(
        y_test,
        y_pred
    )

    # ========================================================
    # CABEÇALHO
    # ========================================================

    print("\n" + "=" * 80)
    print(f"RESULTADOS DO MODELO: {model_name}")
    print("=" * 80)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-Score          : {f1:.4f}")
    print(f"Balanced Accuracy : {balanced_acc:.4f}")
    print(f"MCC               : {mcc:.4f}")
    print(f"Cohen's Kappa     : {kappa:.4f}")

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\n" + "=" * 80)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 80)

    report_text = classification_report(
        y_test,
        y_pred,
        target_names=list(CLASSES.keys()),
        digits=4,
        zero_division=0
    )

    print(report_text)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=list(CLASSES.keys()),
        output_dict=True,
        zero_division=0
    )

    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    class_names = list(CLASSES.keys())

    plt.figure(
        figsize=(11, 8)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        f"{model_name} - Matriz de Confusão"
    )

    plt.xlabel(
        "Classe prevista"
    )

    plt.ylabel(
        "Classe real"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()
    plt.show()

    # ========================================================
    # ANÁLISE COMPLETA POR CLASSE
    # ========================================================

    resultados_por_classe = {}

    print("\n" + "=" * 80)
    print("ANÁLISE DETALHADA POR CLASSE")
    print("=" * 80)

    for i, class_name in enumerate(class_names):

        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP

        sensitivity = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else 0
        )

        specificity = (
            TN / (TN + FP)
            if (TN + FP) > 0
            else 0
        )

        npv = (
            TN / (TN + FN)
            if (TN + FN) > 0
            else 0
        )

        resultados_por_classe[class_name] = {
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "Sensibilidade": float(sensitivity),
            "Especificidade": float(specificity),
            "NPV": float(npv)
        }

        print("\n" + "-" * 80)
        print(f"CLASSE: {class_name}")
        print("-" * 80)

        print(
            f"Acertos para esta classe                  : {TP}"
        )

        print(
            f"Casos desta classe que o modelo perdeu    : {FN}"
        )

        print(
            f"Casos de outras classes classificados aqui: {FP}"
        )

        print(
            f"Casos corretamente rejeitados             : {TN}"
        )

        print(
            f"Sensibilidade                              : "
            f"{sensitivity:.4f} "
            f"({sensitivity * 100:.2f}%)"
        )

        print(
            f"Especificidade                             : "
            f"{specificity:.4f} "
            f"({specificity * 100:.2f}%)"
        )

        print(
            f"Valor Preditivo Negativo (NPV)             : "
            f"{npv:.4f} "
            f"({npv * 100:.2f}%)"
        )

        print("\nINTERPRETAÇÃO:")

        print(
            f"- De todos os casos reais da classe "
            f"'{class_name}', o modelo reconheceu "
            f"{sensitivity * 100:.2f}%."
        )

        print(
            f"- Entre os casos que não pertenciam à classe "
            f"'{class_name}', o modelo rejeitou corretamente "
            f"{specificity * 100:.2f}%."
        )

        print(
            f"- Quando o modelo indicou que uma amostra não era "
            f"da classe '{class_name}', essa decisão esteve correta "
            f"em {npv * 100:.2f}% dos casos."
        )

    # ========================================================
    # DETALHAMENTO DOS ERROS
    # ========================================================

    print("\n" + "=" * 80)
    print(
        "PARA ONDE OS CASOS FORAM CLASSIFICADOS INCORRETAMENTE"
    )
    print("=" * 80)

    interpretacao = {
        "TP": {},
        "FN": {},
        "FP": {},
        "TN": {}
    }

    # ========================================================
    # VERDADEIROS POSITIVOS
    # ========================================================

    print("\nVERDADEIROS POSITIVOS\n")

    for i, class_name in enumerate(class_names):

        TP = cm[i, i]

        interpretacao["TP"][class_name] = int(TP)

        print(
            f"- {class_name}: "
            f"{TP} casos foram identificados corretamente."
        )

    # ========================================================
    # FALSOS NEGATIVOS
    # ========================================================

    print(
        "\nERROS — CASOS REAIS QUE FORAM CONFUNDIDOS\n"
    )

    for i, true_class in enumerate(class_names):

        errors = {}

        for j, predicted_class in enumerate(class_names):

            if i != j and cm[i, j] > 0:

                errors[predicted_class] = int(
                    cm[i, j]
                )

        interpretacao["FN"][true_class] = errors

        print(
            f"\nClasse real: {true_class}"
        )

        if errors:

            for predicted_class, value in errors.items():

                print(
                    f"- {value} caso(s) foram classificados "
                    f"incorretamente como {predicted_class}."
                )

        else:

            print(
                "- Nenhum caso desta classe foi confundido "
                "com outra classe."
            )

    # ========================================================
    # FALSOS POSITIVOS
    # ========================================================

    print(
        "\nERROS — CASOS DE OUTRAS CLASSES "
        "QUE FORAM ATRIBUÍDOS A ESTA\n"
    )

    for j, predicted_class in enumerate(class_names):

        errors = {}

        for i, true_class in enumerate(class_names):

            if i != j and cm[i, j] > 0:

                errors[true_class] = int(
                    cm[i, j]
                )

        interpretacao["FP"][predicted_class] = errors

        print(
            f"\nClasse prevista: {predicted_class}"
        )

        if errors:

            for true_class, value in errors.items():

                print(
                    f"- {value} caso(s) originalmente pertencentes a "
                    f"{true_class} foram classificados como "
                    f"{predicted_class}."
                )

        else:

            print(
                "- Nenhum caso de outra classe foi atribuído "
                "incorretamente a esta classe."
            )

    # ========================================================
    # VERDADEIROS NEGATIVOS
    # ========================================================

    print(
        "\nVERDADEIROS NEGATIVOS\n"
    )

    total = cm.sum()

    for i, class_name in enumerate(class_names):

        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP

        TN = (
            total -
            TP -
            FN -
            FP
        )

        interpretacao["TN"][class_name] = int(
            TN
        )

        print(
            f"- {class_name}: "
            f"{TN} casos de outras classes foram corretamente "
            f"reconhecidos como não pertencentes a esta classe."
        )

    # ========================================================
    # ROC CURVE + AUC
    # ========================================================

    classes_numericas = np.arange(
        len(class_names)
    )

    y_test_bin = label_binarize(
        y_test,
        classes=classes_numericas
    )

    plt.figure(
        figsize=(10, 8)
    )

    auc_scores = []
    auc_por_classe = {}
    roc_por_classe = {}

    print("\n" + "=" * 80)
    print("AUC POR CLASSE")
    print("=" * 80)

    for i, class_name in enumerate(class_names):

        fpr, tpr, thresholds = roc_curve(
            y_test_bin[:, i],
            y_prob[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        auc_scores.append(
            roc_auc
        )

        auc_por_classe[class_name] = float(
            roc_auc
        )

        roc_por_classe[class_name] = {

            "FPR":
                fpr.tolist(),

            "TPR":
                tpr.tolist(),

            "Thresholds":
                thresholds.tolist(),

            "AUC":
                float(roc_auc)

        }

        print(
            f"{class_name:<40}: "
            f"{roc_auc:.4f}"
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{class_name} "
                f"(AUC = {roc_auc:.4f})"
            )
        )

    # ========================================================
    # MÉDIA AUC
    # ========================================================

    mean_auc = np.mean(
        auc_scores
    )

    print("-" * 80)

    print(
        f"{'AUC MÉDIA':<40}: "
        f"{mean_auc:.4f}"
    )

    # ========================================================
    # CURVA ROC
    # ========================================================

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title(
        f"{model_name} - Curva ROC"
    )

    plt.xlabel(
        "Taxa de Falsos Positivos"
    )

    plt.ylabel(
        "Taxa de Verdadeiros Positivos"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("RESUMO FINAL DO MODELO")
    print("=" * 80)

    print(
        f"\nO modelo avaliado foi: {model_name}."
    )

    print(
        f"A acurácia global foi de "
        f"{accuracy * 100:.2f}%."
    )

    print(
        f"O F1-Score global foi de "
        f"{f1 * 100:.2f}%."
    )

    print(
        f"A Balanced Accuracy foi de "
        f"{balanced_acc * 100:.2f}%."
    )

    print(
        f"A AUC média foi de "
        f"{mean_auc:.4f}."
    )

    print(
        "\nDesempenho individual das classes:"
    )

    for class_name in class_names:

        dados = resultados_por_classe[
            class_name
        ]

        print(
            f"\n- {class_name}:"
        )

        print(
            f"  Sensibilidade: "
            f"{dados['Sensibilidade'] * 100:.2f}%"
        )

        print(
            f"  Especificidade: "
            f"{dados['Especificidade'] * 100:.2f}%"
        )

        print(
            f"  NPV: "
            f"{dados['NPV'] * 100:.2f}%"
        )

    # ========================================================
    # RETORNO COMPLETO
    # ========================================================

    resultados = {

        "Modelo":
            model_name,

        "Accuracy":
            float(accuracy),

        "Precision":
            float(precision),

        "Recall":
            float(recall),

        "F1-Score":
            float(f1),

        "Balanced Accuracy":
            float(balanced_acc),

        "MCC":
            float(mcc),

        "Cohen's Kappa":
            float(kappa),

        "AUC por Classe":
            auc_por_classe,

        "Mean AUC":
            float(mean_auc),

        "Curvas ROC":
            roc_por_classe,

        "Por Classe":
            resultados_por_classe,

        "Interpretacao":
            interpretacao,

        "Classification Report":
            report_dict,

        "Matriz de Confusao":
            cm,

        "y_test":
            y_test,

        "y_pred":
            y_pred,

        "y_prob":
            y_prob

    }

    return resultados


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("=" * 80)
    print(
        "TREINAMENTO E AVALIAÇÃO - LIGHTGBM"
    )
    print("=" * 80)

    print(
        "\nDataset path:"
    )

    print(
        DATA_ROOT
    )

    # ========================================================
    # 1. CARREGA O DATASET
    # ========================================================

    all_paths, all_labels, CLASSES = load_dataset(
        DATA_ROOT
    )

    print(
        f"\nTotal samples: {len(all_paths)}"
    )

    # ========================================================
    # 2. DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        {
            "image_path":
                all_paths,

            "label":
                all_labels
        }
    )

    print(
        "\nPrimeiras amostras:"
    )

    print(
        df.head()
    )

    # ========================================================
    # 3. DISTRIBUIÇÃO DAS CLASSES
    # ========================================================

    class_counts = Counter(
        all_labels
    )

    class_names = list(
        CLASSES.keys()
    )

    counts = [
        class_counts[i]

        for i in range(
            len(class_names)
        )
    ]

    print(
        "\nDistribuição das classes:"
    )

    for class_name, count in zip(
        class_names,
        counts
    ):

        print(
            f"{class_name}: {count}"
        )

    total_amostras = len(
        all_labels
    )

    distribuicao_classes = {

        class_name: {

            "Quantidade":
                int(count),

            "Percentual":
                float(
                    (
                        count /
                        total_amostras
                    ) *
                    100
                )

        }

        for class_name, count in zip(
            class_names,
            counts
        )

    }

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        class_names,
        counts
    )

    plt.title(
        "Distribuição das Classes ECG"
    )

    plt.xlabel(
        "Classes"
    )

    plt.ylabel(
        "Quantidade de Imagens"
    )

    plt.xticks(
        rotation=15
    )

    plt.tight_layout()
    plt.show()

    plt.figure(
        figsize=(8, 8)
    )

    plt.pie(
        counts,
        labels=class_names,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(
        "Distribuição Percentual das Classes ECG"
    )

    plt.axis(
        "equal"
    )

    plt.show()

    # ========================================================
    # 4. DIVISÃO DOS DADOS
    # TREINO = 50%
    # VALIDAÇÃO = 25%
    # TESTE = 25%
    # ========================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_paths,
        all_labels,
        test_size=0.50,
        stratify=all_labels,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print(
        "\nDivisão dos dados:"
    )

    print(
        f"Treino    : {len(X_train)}"
    )

    print(
        f"Validação : {len(X_val)}"
    )

    print(
        f"Teste     : {len(X_test)}"
    )

    divisao_dados = {

        "Total":
            int(
                len(all_paths)
            ),

        "Treino": {

            "Quantidade":
                int(
                    len(X_train)
                ),

            "Percentual":
                float(
                    len(X_train) /
                    len(all_paths) *
                    100
                )

        },

        "Validacao": {

            "Quantidade":
                int(
                    len(X_val)
                ),

            "Percentual":
                float(
                    len(X_val) /
                    len(all_paths) *
                    100
                )

        },

        "Teste": {

            "Quantidade":
                int(
                    len(X_test)
                ),

            "Percentual":
                float(
                    len(X_test) /
                    len(all_paths) *
                    100
                )

        }

    }

    # ========================================================
    # 5. PRÉ-PROCESSAMENTO
    # ========================================================

    print(
        "\nProcessando imagens de treino..."
    )

    X_train_processed = np.array(
        [
            process_image(path)

            for path in X_train
        ]
    )

    print(
        "Processando imagens de validação..."
    )

    X_val_processed = np.array(
        [
            process_image(path)

            for path in X_val
        ]
    )

    print(
        "Processando imagens de teste..."
    )

    X_test_processed = np.array(
        [
            process_image(path)

            for path in X_test
        ]
    )

    # ========================================================
    # 6. NORMALIZAÇÃO / PADRONIZAÇÃO
    # ========================================================

    scaler = StandardScaler()

    X_train_processed = scaler.fit_transform(
        X_train_processed
    )

    X_val_processed = scaler.transform(
        X_val_processed
    )

    X_test_processed = scaler.transform(
        X_test_processed
    )

    # ========================================================
    # 7. LIGHTGBM
    # ========================================================

    lgbm_model = LGBMClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    # ========================================================
    # 8. TREINA + AVALIA
    # ========================================================

    resultado_lgbm = evaluate_model(
        "LightGBM",
        lgbm_model,
        X_train_processed,
        y_train,
        X_test_processed,
        y_test,
        CLASSES
    )

    # ========================================================
    # 9. SALVA O PACOTE COMPLETO EM .PKL
    # ========================================================

    pacote_modelo = {

        "modelo":
            lgbm_model,

        "scaler":
            scaler,

        "classes":
            CLASSES,

        "image_size":
            IMAGE_SIZE,

        "grayscale":
            True,

        "normalizacao_pixels":
            "img / 255.0",

        "flatten":
            True,

        "random_state":
            RANDOM_STATE,

        "dataset": {

            "Total de Amostras":
                int(
                    len(all_paths)
                ),

            "Distribuicao das Classes":
                distribuicao_classes,

            "Divisao dos Dados":
                divisao_dados

        },

        "resultado_teste":
            resultado_lgbm

    }


    # ========================================================
    # SALVA EM dump/LightGBM-model.pkl
    # ========================================================

    joblib.dump(
        pacote_modelo,
        MODEL_OUTPUT,
        compress=3
    )


    print(
        "\n" + "=" * 80
    )

    print(
        "MODELO SALVO COM SUCESSO"
    )

    print(
        "=" * 80
    )


    print(
        f"\nArquivo gerado:\n{MODEL_OUTPUT}"
    )


    print(
        "\nO arquivo .pkl contém:"
    )

    print(
        "- modelo LightGBM treinado"
    )

    print(
        "- StandardScaler ajustado"
    )

    print(
        "- mapeamento das classes"
    )

    print(
        "- tamanho de entrada 128x128"
    )

    print(
        "- configuração de pré-processamento"
    )

    print(
        "- métricas obtidas no conjunto de teste"
    )


# ============================================================
# INÍCIO
# ============================================================

if __name__ == "__main__":

    main()