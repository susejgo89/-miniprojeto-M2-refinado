# 📌 Quadro Kanban do Projeto: IA para Análise Preditiva & Deploy de API REST (MNIST)

Este documento registra o planejamento, a rota de desenvolvimento, o status de cada tarefa e o histórico de commits do projeto, garantindo total rastreabilidade do processo de engenharia de software, ciência de dados e MLOps.

---

## 📊 Visão Geral do Tabuleiro

### 🔹 Ciclo 1: Pesquisa, Modelagem e Avaliação Experimental (Semanas 05 a 12)
| Tarefa | Status |
| :--- | :--- |
| **Passo 1**: Setup Inicial, Ambiente & Git | ✅ Concluído |
| **Passo 2**: Fase 1 - EDA & Carga MNIST | ✅ Concluído | 
| **Passo 3**: Fase 2 - Split Estratificado & Normalização | ✅ Concluído |
| **Passo 4**: Fase 3 - Modelagem (SVM, RF, LightGBM, MLP) | ✅ Concluído | 
| **Passo 5**: Fase 4 - Matrizes de Confusão & Benchmark | ✅ Concluído |
| **Passo 6**: Fase 5.1/5.2 - Testes OOD & Falsa Certeza | ✅ Concluído | 
| **Passo 7**: Fase 5.3 - PDI com OpenCV & Streamlit | ✅ Concluído |
| **Passo 8**: Documentação Inicial (README Ciclo 1) | ✅ Concluído |

### 🔹 Ciclo 2: Projeto Avaliativo Final — Microsserviço Preditivo com FastAPI (Semana 13)
| 📋 Backlog | ⏳ Em Progresso | 🔍 Em Revisão / Validação | ✅ Concluído |
| :--- | :--- | :--- | :--- |
| **Passo 15**: GitFlow Final (Merge develop -> main) | **Passo 14**: Roteiro Vídeo (Item 5.4) | | **Passo 9**: Fase 1 - Verificação do Modelo & Métricas |
| | | | **Passo 10**: Fase 2 - Estrutura Modular da API |
| | | | **Passo 11**: Fase 3 - Ciclo de Vida (Lifespan RAM) |
| | | | **Passo 12**: Fase 4 - Validação Estrita (Pydantic) |
| | | | **Passo 13**: Fase 5 - Endpoints (`/health`, `/predict`) |

---

## 🗺️ Detalhamento das Etapas e Checklist de Entregas

### [x] Passo 1: Inicialização do Repositório, Ambiente & Estrutura
- [x] Criar estrutura de diretórios (`data/custom_digits/`, `notebooks/`, `src/`).
- [x] Configurar `.gitignore` e `requirements.txt`.
- [x] Inicializar Git (`main`), primeiro commit e criação da branch `develop`.
- [x] Criar ambiente virtual `.venv` e instalar dependências.
- [x] Registrar kernel do Jupyter (`miniprojeto-m2`).
- [x] Criar arquivo `KANBAN.md` de acompanhamento.
- **Branch**: `main` -> `develop`
- **Commit**: `configura estrutura inicial do projeto e dependencias`

---

### [x] Passo 2: Fase 1 - Carregamento e Análise Exploratória de Imagens (EDA)
- [x] Criar branch `feature/fase1-eda` a partir de `develop`.
- [x] Criar notebook `notebooks/mnist_pipeline.ipynb` com cabeçalhos estruturados.
- [x] Carregar dataset MNIST (`fetch_openml('mnist_784')`).
- [x] Analisar dimensões das matrizes $X$ (70.000, 784) e $y$ (70.000).
- [x] Verificar e comprovar o balanceamento de classes (dígitos de 0 a 9) com gráfico e tabela.
- [x] Construir grade visual ($2 \times 5$) com `matplotlib` exibindo amostras reais de cada dígito com seus rótulos.
- [x] Célula conceitual: Explicar detalhadamente a escala de intensidade de pixels (0 a 255) e a representação vetorial ($28 \times 28 = 784$ features).
- **Branch**: `feature/fase1-eda`
- **Commit**: `implementa analise exploratoria e carregamento do mnist`

---

### [x] Passo 3: Fase 2 - Pipeline de Pré-processamento e Divisão dos Dados
- [x] Criar branch `feature/fase2-preprocessamento` a partir de `develop`.
- [x] Divisão estratificada (`stratify=y`) em Treino, Validação e Teste (ex: 70/10/20 ou 80/20).
- [x] Normalização dos pixels para o intervalo $[0.0, 1.0]$ dividindo por 255.0.
- [x] Célula conceitual: Justificar textualmente a importância da normalização para convergência de modelos lineares e distâncias métricas.
- **Branch**: `feature/fase2-preprocessamento`
- **Commit**: `implementa split estratificado e normalizacao de dados`

---

### [x] Passo 4: Fase 3 - Implementação e Treinamento de 3 Modelos Distintos
- [x] Criar branch `feature/fase3-modelagem` a partir de `develop`.
- [x] Implementar Modelo 1: Random Forest Classifier (Ajuste de `n_estimators` e `max_depth`).
- [x] Implementar Modelo 2: LightGBM — Microsoft (Ajuste de `learning_rate`).
- [x] Implementar Modelo 3: Support Vector Machine (SVM) (Ajuste de `C` e `kernel`).
- [x] Implementar Modelo 4: Multi-Layer Perceptron (MLP) (Ajuste de neurônios, taxa de aprendizado e ativação).
- [x] Registrar tempo de treinamento de cada modelo.
- **Branch**: `feature/fase3-modelagem`
- **Commit**: `implementa e treina modelos random forest, svm e mlp`

---

### [x] Passo 5: Fase 4 - Avaliação Comparativa de Desempenho
- [x] Criar branch `feature/fase4-avaliacao-comparativa` a partir de `develop`.
- [x] Gerar Matrizes de Confusão completas ($10 \times 10$) em Heatmap para os 3 modelos.
- [x] Gerar Tabela Comparativa Consolidada (Accuracy, Precision ponderada, Recall ponderado, F1-Score ponderado).
- [x] Célula técnica de conclusão: Análise de dígitos mais confusos (ex: 4 vs 9, 3 vs 5), melhor modelo e custo computacional (tempo vs acurácia).
- **Branch**: `feature/fase4-avaliacao-comparativa`
- **Commit**: `implementa avaliacao comparativa e matrizes de confusao`

---

### [x] Passo 6: Fase 5.1 & 5.2 - Mascaramento de Classes & Generalização Extrema (OOD)
- [x] Criar branch `feature/fase5-generalizacao-ood` a partir de `develop`.
- [x] Desafio A: Remover pelo menos duas classes do treino (ex: dígitos 4 e 7) e treinar modelo sem vê-las.
- [x] Desafio B: Testar o modelo exclusivamente nas classes ocultadas.
- [x] Plotar distribuição de predições, matriz de confusão e analisar o fenômeno de "Falsa Certeza" (Overconfidence).
- **Branch**: `feature/fase5-generalizacao-ood`
- **Commit**: `implementa testes de estresse com mascaramento e inferencia ood`

---

### [x] Passo 7: Fase 5.3 - Desafio C: Inferência com Imagens Manuscritas Próprias
- [x] Criar branch `feature/fase5-inferencia-customizada` a partir de `develop`.
- [x] Criar pipeline de visão computacional (OpenCV/Pillow): Grayscale -> Inversão -> Bounding Box -> Redimensionamento $28 \times 28$ com centro de massa -> Normalização $[0, 1]$.
- [x] Salvar amostras de imagens reais em `data/custom_digits/`.
- [x] Realizar predição com o melhor modelo e plotar imagem processada lado a lado com gráfico de probabilidades.
- **Branch**: `feature/fase5-inferencia-customizada`
- **Commit**: `implementa pipeline de visao computacional para digitos manuscritos reais`

---

### [x] Passo 8: Documentação Inicial (README.md) & Aplicação Streamlit
- [x] Criar branch `feature/documentacao-readme` a partir de `develop`.
- [x] Criar `README.md` completo com escopo, arquitetura, instruções de execução do ciclo 1 e resultados.
- [x] Implementar aplicação web interativa em `src/app.py` (Streamlit com canvas de desenho).
- **Branch**: `feature/documentacao-readme` -> `develop` -> `main`
- **Commit**: `adiciona documentacao completa do projeto no readme`

---

### 🚀 CICLO 2: MÓDULO 2 - PROJETO AVALIATIVO FINAL (SEMANA 13)

### [x] Passo 9: Fase 1 (API) - Preparação e Serialização do Modelo Preditivo
- [x] Selecionar a **Opção A**: Reaproveitamento do modelo campeão de Reconhecimento de Dígitos Manuscritos (MNIST).
- [x] Confirmar artefatos serializados em `models/` (`modelo_mlp_mnist.keras` e `modelo_mlp_mnist.joblib`).
- [x] Documentar métricas de avaliação alcançadas pelo modelo (Acurácia: 97.39%, F1-Score: 97.39%, Precision: 97.39%, Recall: 97.39%).
- **Branch**: `develop` -> `feature/estrutura-api-fastapi`
- **Status**: ✅ Concluído

---

### [x] Passo 10: Fase 2 (API) - Estrutura Modular da API FastAPI
- [x] Criar pacote da API desacoplado em camadas (`schemas/`, `services/`, `routers/` e `main.py`).
- [x] Isolar responsabilidades: desacoplar protocolos de rede HTTP das operações matemáticas de Machine Learning.
- [x] Atualizar `requirements.txt` com dependências de produção (`fastapi`, `uvicorn`, `pydantic`, `httpx`).
- **Branch**: `feature/estrutura-api-fastapi`
- **Commit**: `implementa estrutura modular da api fastapi`
- **Status**: ✅ Concluído

---

### [x] Passo 11: Fase 3 (API) - Ciclo de Vida e Carregamento Otimizado (Lifespan)
- [x] Implementar `@asynccontextmanager` com gerenciador de contexto `lifespan` no FastAPI.
- [x] Carregar o artefato do modelo em memória RAM **uma única vez** durante o startup da aplicação.
- [x] Garantir conformidade com a proibição técnica do edital: **não ler o disco durante requisições preditivas**.
- [x] Disponibilizar o modelo carregado no estado global da aplicação (`app.state.predictor_service`).
- **Branch**: `feature/estrutura-api-fastapi`
- **Commit**: `implementa ciclo de vida lifespan para carregamento do modelo em memoria ram`
- **Status**: ✅ Concluído

---

### [x] Passo 12: Fase 4 (API) - Modelagem e Validação Estrita de Dados (Pydantic)
- [x] Criar esquemas em `schemas/prediction.py` e `schemas/health.py` herdando de `pydantic.BaseModel`.
- [x] Configurar tipagem estática e restrições detalhadas com `Field` (vetor de 784 pixels, limites com `ge=0.0`, `le=1.0` ou escala 0 a 255).
- [x] Implementar esquema de resposta tipada (`prediction`, `confidence`, `probabilities`, `inference_time_ms`).
- [x] Validar retorno automático de **HTTP 422 Unprocessable Entity** para dados de entrada corrompidos ou fora dos limites.
- **Branch**: `feature/estrutura-api-fastapi`
- **Commit**: `implementa validacao estrita de dados com schemas pydantic`
- **Status**: ✅ Concluído

---

### [x] Passo 13: Fase 5 (API) - Implementação de Endpoints e Tratamento de Erros
- [x] Criar roteador de monitoramento: `GET /health` respondendo HTTP 200 OK com status de integridade e presença do modelo na RAM.
- [x] Criar roteador de inferência: `POST /predict` (e `/v1/predict`) consumindo o schema validado, invocando o service e devolvendo HTTP 200 OK.
- [x] Implementar tratamento semântico de erros com `HTTPException` e fallbacks (HTTP 200, 400, 422, 500) evitando qualquer crash na aplicação.
- **Branch**: `feature/estrutura-api-fastapi`
- **Commit**: `implementa endpoints health e predict com tratamento semantico de erros`
- **Status**: ✅ Concluído

---

### [ ] Passo 14: Documentação no README.md & Roteiro de Gravação do Vídeo
- [ ] Atualizar `README.md` contendo todos os 6 tópicos obrigatórios do item 5.2 do edital:
  1. Nome e Resumo do Projeto (problema de negócio).
  2. Origem do Modelo (indicação da base MNIST do Módulo 2).
  3. Tecnologias Utilizadas e versões exatas.
  4. Instruções de Instalação e Execução (`venv`, `requirements.txt`, `uvicorn`).
  5. Documentação da API com payloads JSON de exemplo e Swagger `/docs`.
  6. Seção para link do vídeo de evidência no Google Drive.
- [ ] Elaborar script completo para o vídeo de até 10 minutos respondendo ponto a ponto as 6 perguntas do item 5.4.
- **Branch**: `feature/estrutura-api-fastapi`
- **Commit Planejado**: `atualiza readme com documentacao da api rest e roteiro de apresentacao`
- **Status**: 📋 Backlog

---

### [ ] Passo 15: Conclusão GitFlow e Merges Finais
- [ ] Executar bateria de testes com `pytest` e `httpx` cobrindo `/health`, `/predict` (válido) e `/predict` (inválido 422).
- [ ] Realizar merge da branch `feature/estrutura-api-fastapi` para a branch `develop`.
- [ ] Realizar merge da branch `develop` para a branch `main` mantendo histórico completo e branches preservadas.
- **Branch**: `feature/estrutura-api-fastapi` -> `develop` -> `main`
- **Commit Planejado**: `merge: integra microsservico fastapi na branch principal`
- **Status**: 📋 Backlog

---

## 📜 Histórico de Commits e Rastreabilidade

| Data / Hora | Branch | Mensagem do Commit | Descrição da Entrega |
| :--- | :--- | :--- | :--- |
| 31/08/2026 | `main` | `configura estrutura inicial do projeto e dependencias` | Estrutura base, `.gitignore` e `requirements.txt`. |
| 12/09/2026 | `main` | `feat: versao inicial do pipeline mnist refinado` | Notebooks, modelos treinados e assets do Ciclo 1. |
| 26/10/2026 | `develop` | `cria branch develop a partir da main` | Inicialização do fluxo GitFlow para o projeto final (Semana 13). |
| 26/10/2026 | `feature/estrutura-api-fastapi` | `atualiza kanban com planejamento do microsservico fastapi` | Planejamento completo dos passos 9 a 15 do Ciclo 2. |


