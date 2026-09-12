document.addEventListener(
    "DOMContentLoaded",
    () => {

        // ============================================================
        // ELEMENTOS
        // ============================================================

        const uploadArea =
            document.getElementById(
                "uploadArea"
            );

        const inputImagem =
            document.getElementById(
                "imagem"
            );

        const previewContainer =
            document.getElementById(
                "previewContainer"
            );

        const preview =
            document.getElementById(
                "preview"
            );

        const botaoAnalisar =
            document.getElementById(
                "analisar"
            );

        const loading =
            document.getElementById(
                "loading"
            );

        const resultado =
            document.getElementById(
                "resultado"
            );

        const erro =
            document.getElementById(
                "erro"
            );

        const classeResultado =
            document.getElementById(
                "classeResultado"
            );

        const confiancaValor =
            document.getElementById(
                "confiancaValor"
            );

        const barraConfianca =
            document.getElementById(
                "barraConfianca"
            );

        const probabilidades =
            document.getElementById(
                "probabilidades"
            );


        // ============================================================
        // ARQUIVO SELECIONADO
        // ============================================================

        let arquivoSelecionado = null;


        // ============================================================
        // MOSTRAR ELEMENTO
        // ============================================================

        function mostrar(
            elemento
        ) {

            if (elemento) {

                elemento.classList.remove(
                    "escondido"
                );

            }

        }


        // ============================================================
        // ESCONDER ELEMENTO
        // ============================================================

        function esconder(
            elemento
        ) {

            if (elemento) {

                elemento.classList.add(
                    "escondido"
                );

            }

        }


        // ============================================================
        // LIMPAR ERRO
        // ============================================================

        function limparErro() {

            if (!erro) {
                return;
            }

            erro.textContent = "";

            esconder(
                erro
            );

        }


        // ============================================================
        // MOSTRAR ERRO
        // ============================================================

        function mostrarErro(
            mensagem
        ) {

            if (!erro) {
                return;
            }

            erro.textContent =
                mensagem;

            mostrar(
                erro
            );

        }


        // ============================================================
        // LIMPAR RESULTADO
        // ============================================================

        function limparResultado() {

            esconder(
                resultado
            );


            classeResultado.textContent =
                "-";


            confiancaValor.textContent =
                "0%";


            barraConfianca.style.width =
                "0%";


            probabilidades.innerHTML =
                "";

        }


        // ============================================================
        // VALIDAÇÃO DA IMAGEM
        // ============================================================

        function validarArquivo(
            arquivo
        ) {

            if (!arquivo) {

                mostrarErro(
                    "Nenhuma imagem foi selecionada."
                );

                return false;

            }


            // --------------------------------------------------------
            // VALIDA PELA EXTENSÃO
            // --------------------------------------------------------

            const nome =
                arquivo.name.toLowerCase();


            const extensaoValida =
                nome.endsWith(".png") ||
                nome.endsWith(".jpg") ||
                nome.endsWith(".jpeg");


            if (!extensaoValida) {

                mostrarErro(
                    "Formato não permitido. Utilize PNG, JPG ou JPEG."
                );

                return false;

            }


            // --------------------------------------------------------
            // LIMITE 10 MB
            // --------------------------------------------------------

            const limite =
                10 *
                1024 *
                1024;


            if (
                arquivo.size >
                limite
            ) {

                mostrarErro(
                    "A imagem ultrapassa o limite de 10 MB."
                );

                return false;

            }


            return true;

        }


        // ============================================================
        // SELEÇÃO DA IMAGEM
        // ============================================================

        function selecionarImagem(
            arquivo
        ) {

            limparErro();

            limparResultado();


            if (
                !validarArquivo(
                    arquivo
                )
            ) {

                arquivoSelecionado =
                    null;

                botaoAnalisar.disabled =
                    true;

                return;

            }


            // ========================================================
            // GUARDA IMAGEM
            // ========================================================

            arquivoSelecionado =
                arquivo;


            // ========================================================
            // LIBERA BOTÃO ANALISAR
            // ========================================================

            botaoAnalisar.disabled =
                false;


            // ========================================================
            // PREVIEW
            // ========================================================

            const leitor =
                new FileReader();


            leitor.onload =
                (
                    evento
                ) => {

                    preview.src =
                        evento
                            .target
                            .result;


                    mostrar(
                        previewContainer
                    );

                };


            leitor.readAsDataURL(
                arquivo
            );

        }


        // ============================================================
        // INPUT DA IMAGEM
        // ============================================================

        inputImagem.addEventListener(
            "change",
            (
                evento
            ) => {

                const arquivo =
                    evento
                        .target
                        .files[0];


                selecionarImagem(
                    arquivo
                );

            }
        );


        // ============================================================
        // ARRASTAR SOBRE A ÁREA
        // ============================================================

        uploadArea.addEventListener(
            "dragover",
            (
                evento
            ) => {

                evento.preventDefault();


                uploadArea
                    .classList
                    .add(
                        "arrastando"
                    );

            }
        );


        // ============================================================
        // SAIR DA ÁREA
        // ============================================================

        uploadArea.addEventListener(
            "dragleave",
            () => {

                uploadArea
                    .classList
                    .remove(
                        "arrastando"
                    );

            }
        );


        // ============================================================
        // SOLTAR IMAGEM
        // ============================================================

        uploadArea.addEventListener(
            "drop",
            (
                evento
            ) => {

                evento.preventDefault();


                uploadArea
                    .classList
                    .remove(
                        "arrastando"
                    );


                const arquivo =
                    evento
                        .dataTransfer
                        .files[0];


                selecionarImagem(
                    arquivo
                );

            }
        );


        // ============================================================
        // PROBABILIDADES
        // ============================================================

        function renderizarProbabilidades(
            dados
        ) {

            probabilidades.innerHTML =
                "";


            const lista =
                Object.entries(
                    dados || {}
                );


            lista.sort(
                (
                    a,
                    b
                ) =>
                    b[1] -
                    a[1]
            );


            lista.forEach(
                (
                    [
                        classe,
                        valor
                    ]
                ) => {

                    const item =
                        document.createElement(
                            "div"
                        );


                    item.className =
                        "probabilidade-item";


                    const linha =
                        document.createElement(
                            "div"
                        );


                    linha.className =
                        "linha-titulo";


                    const nome =
                        document.createElement(
                            "span"
                        );


                    nome.textContent =
                        classe;


                    const percentual =
                        document.createElement(
                            "strong"
                        );


                    percentual.textContent =
                        `${Number(valor).toFixed(2)}%`;


                    linha.appendChild(
                        nome
                    );


                    linha.appendChild(
                        percentual
                    );


                    const barra =
                        document.createElement(
                            "div"
                        );


                    barra.className =
                        "barra";


                    const preenchimento =
                        document.createElement(
                            "div"
                        );


                    preenchimento.className =
                        "barra-preenchida";


                    preenchimento.style.width =
                        `${Number(valor)}%`;


                    barra.appendChild(
                        preenchimento
                    );


                    item.appendChild(
                        linha
                    );


                    item.appendChild(
                        barra
                    );


                    probabilidades.appendChild(
                        item
                    );

                }
            );

        }


        // ============================================================
        // BOTÃO ANALISAR
        // ============================================================

        botaoAnalisar.addEventListener(
            "click",
            async () => {

                // ----------------------------------------------------
                // VERIFICA IMAGEM
                // ----------------------------------------------------

                if (!arquivoSelecionado) {

                    mostrarErro(
                        "Selecione uma imagem de ECG."
                    );

                    return;

                }


                limparErro();

                limparResultado();


                mostrar(
                    loading
                );


                botaoAnalisar.disabled =
                    true;


                // ====================================================
                // FORM DATA
                // ====================================================

                const formData =
                    new FormData();


                formData.append(
                    "imagem",
                    arquivoSelecionado
                );


                try {

                    // =================================================
                    // ENVIA PARA O FLASK
                    // =================================================

                    const resposta =
                        await fetch(
                            "/ecg/predict",
                            {

                                method:
                                    "POST",

                                body:
                                    formData

                            }
                        );


                    const dados =
                        await resposta.json();


                    // =================================================
                    // ERRO DO FLASK
                    // =================================================

                    if (!resposta.ok) {

                        throw new Error(
                            dados.erro ||
                            "Não foi possível analisar a imagem."
                        );

                    }


                    // =================================================
                    // CLASSE
                    // =================================================

                    classeResultado.textContent =
                        dados.classe;


                    // =================================================
                    // CONFIANÇA
                    // =================================================

                    const confianca =
                        Number(
                            dados.confianca
                        );


                    confiancaValor.textContent =
                        `${confianca.toFixed(2)}%`;


                    barraConfianca.style.width =
                        `${confianca}%`;


                    // =================================================
                    // PROBABILIDADES
                    // =================================================

                    renderizarProbabilidades(
                        dados.probabilidades
                    );


                    // =================================================
                    // MOSTRA RESULTADO
                    // =================================================

                    mostrar(
                        resultado
                    );

                }

                catch (
                    erroRequisicao
                ) {

                    mostrarErro(
                        erroRequisicao.message ||
                        "Erro ao realizar a classificação."
                    );

                }

                finally {

                    esconder(
                        loading
                    );


                    // =================================================
                    // LIBERA NOVAMENTE O BOTÃO
                    // =================================================

                    botaoAnalisar.disabled =
                        false;

                }

            }
        );

    }
);