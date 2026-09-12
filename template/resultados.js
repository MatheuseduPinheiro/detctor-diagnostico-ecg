document.addEventListener(
    "DOMContentLoaded",
    function () {

        // ============================================================
        // CLASSE ESCOLHIDA PELO MODELO
        // ============================================================

        const elementoClasseEscolhida =
            document.getElementById(
                "valorClassePredita"
            );


        if (!elementoClasseEscolhida) {

            return;

        }


        const classeEscolhida =
            elementoClasseEscolhida
                .textContent
                .trim();


        // ============================================================
        // CARDS DAS QUATRO CLASSES
        // ============================================================

        const cardsClasses =
            document.querySelectorAll(
                ".classe-metrica-card"
            );


        if (
            cardsClasses.length === 0
        ) {

            return;

        }


        // ============================================================
        // PERCORRE AS QUATRO CLASSES
        // ============================================================

        cardsClasses.forEach(
            function (
                card,
                indice
            ) {

                // ====================================================
                // NOME DA CLASSE DO CARD
                // ====================================================

                const nomeClasse =
                    card
                        .dataset
                        .nomeClasse
                        ?.trim();


                // ====================================================
                // ID INDIVIDUAL PARA MANIPULAÇÃO DO DOM
                // ====================================================

                card.id =
                    "classeMetrica" +
                    (
                        indice + 1
                    );


                // ====================================================
                // REMOVE ESTADO ANTERIOR
                // ====================================================

                card.classList.remove(
                    "classe-metrica-selecionada"
                );


                card.classList.remove(
                    "classe-metrica-nao-selecionada"
                );


                // ====================================================
                // COMPARA COM A CLASSIFICAÇÃO DO MODELO
                // ====================================================

                if (
                    nomeClasse ===
                    classeEscolhida
                ) {

                    // ================================================
                    // CLASSE ESCOLHIDA
                    // ================================================

                    card.classList.add(
                        "classe-metrica-selecionada"
                    );


                    card.setAttribute(
                        "data-selecionada",
                        "true"
                    );


                    card.setAttribute(
                        "aria-label",
                        nomeClasse +
                        " - classe escolhida pelo modelo"
                    );


                    // ================================================
                    // TÍTULO DA CLASSE
                    // ================================================

                    const titulo =
                        card.querySelector(
                            ".classe-metrica-titulo"
                        );


                    if (titulo) {

                        titulo.classList.add(
                            "classe-metrica-titulo-selecionado"
                        );

                    }


                    // ================================================
                    // NOMES DAS MÉTRICAS
                    // ================================================

                    const labels =
                        card.querySelectorAll(
                            ".metrica-clinica-label"
                        );


                    labels.forEach(
                        function (label) {

                            label.classList.add(
                                "metrica-clinica-label-selecionada"
                            );

                        }
                    );


                    // ================================================
                    // VALORES DAS MÉTRICAS
                    // ================================================

                    const valores =
                        card.querySelectorAll(
                            ".metrica-clinica-valor"
                        );


                    valores.forEach(
                        function (valor) {

                            valor.classList.add(
                                "metrica-clinica-valor-selecionada"
                            );

                        }
                    );


                    // ================================================
                    // MARCADOR VISUAL
                    // ================================================

                    const marcador =
                        document.createElement(
                            "span"
                        );


                    marcador.className =
                        "classe-metrica-marcador";


                    marcador.textContent =
                        "Classe selecionada pelo modelo";


                    marcador.setAttribute(
                        "aria-hidden",
                        "true"
                    );


                    if (
                        titulo &&
                        !card.querySelector(
                            ".classe-metrica-marcador"
                        )
                    ) {

                        titulo.insertAdjacentElement(
                            "afterend",
                            marcador
                        );

                    }

                }

                else {

                    // ================================================
                    // DEMAIS CLASSES
                    // ================================================

                    card.classList.add(
                        "classe-metrica-nao-selecionada"
                    );


                    card.setAttribute(
                        "data-selecionada",
                        "false"
                    );

                }

            }
        );


        // ============================================================
        // INTERPRETAÇÃO DA MATRIZ DE CONFUSÃO
        // ============================================================

        const classeMatriz =
            document.getElementById(
                "classeEscolhidaMatriz"
            );


        if (classeMatriz) {

            classeMatriz.classList.add(
                "classe-escolhida-matriz-ativa"
            );

        }


        // ============================================================
        // VERDADEIROS POSITIVOS
        // ============================================================

        const verdadeirosPositivos =
            document.getElementById(
                "matrizVerdadeirosPositivos"
            );


        if (verdadeirosPositivos) {

            verdadeirosPositivos.classList.add(
                "interpretacao-matriz-escolhido"
            );

        }

    }
);