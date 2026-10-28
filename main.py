"""
Ponto de Entrada Oficial da API REST (FastAPI).
Atende às Fases 2, 3 e 5:
- Arquitetura modular e inclusão de roteadores.
- Ciclo de vida Lifespan otimizado (@asynccontextmanager) carregando o modelo em RAM.
- Documentação interativa Swagger (/docs).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api.services.predictor import MNISTPredictorService
from api.routers.health import router as health_router
from api.routers.predict import router as predict_router


# ==============================================================================
# 🧠 1. CICLO DE VIDA OTIMIZADO (LIFESPAN) - FASE 3
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerenciador de Ciclo de Vida da Aplicação.
    Garante que o modelo seja carregado na memória RAM uma única vez na inicialização,
    eliminando gargalos de I/O em disco a cada requisição preditiva.
    """
    print("\n" + "=" * 60)
    print("🚀 [STARTUP] Inicializando microsserviço preditivo MNIST...")
    
    # 1. Instanciamos o serviço apontando para o modelo campeão Keras
    predictor = MNISTPredictorService(model_path="models/modelo_mlp_mnist.keras")
    
    # 2. Carregamento em memória RAM (operação única)
    try:
        predictor.load_model()
        print(f"✅ [STARTUP] Modelo Keras carregado com sucesso na RAM!")
        print(f"📟 [STARTUP] Dispositivo alocado: {predictor.device}")
    except Exception as e:
        print(f"❌ [STARTUP] ERRO CRÍTICO ao carregar o modelo: {e}")

    # 3. Alocamos o serviço no estado global da aplicação (RAM)
    app.state.predictor_service = predictor
    print("=" * 60 + "\n")

    # A aplicação fica viva e pronta para atender requisições
    yield

    # ==========================================================================
    # 🛑 SHUTDOWN (Encerramento Gracioso)
    # ==========================================================================
    print("\n🛑 [SHUTDOWN] Encerrando microsserviço e liberando recursos...")
    app.state.predictor_service = None
    print("👋 [SHUTDOWN] Aplicação finalizada com sucesso.\n")


# ==============================================================================
# 🌐 2. INICIALIZAÇÃO DA APLICAÇÃO FASTAPI
# ==============================================================================
app = FastAPI(
    title="MNIST Vision AI — API REST Preditiva",
    description=(
        "Microsserviço de alta performance para inferência em tempo real de dígitos manuscritos (0 a 9). "
        "Desenvolvido com FastAPI, validação estrita via Pydantic e modelo de Rede Neural MLP em memória RAM."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Interface Swagger UI interativa
    redoc_url="/redoc"     # Documentação técnica alternativa
)


# ==============================================================================
# 🛡️ 3. CONFIGURAÇÃO DE CORS (Cross-Origin Resource Sharing)
# ==============================================================================
# Permite que aplicações frontend (web/mobile) consumam esta API sem bloqueios de navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 🚪 4. MONTAGEM DOS ROTEADORES MODULARES (FASE 2 E 5)
# ==============================================================================
app.include_router(health_router)
app.include_router(predict_router)


# ==============================================================================
# 🏠 5. ROTA RAIZ AMIGÁVEL (GET /)
# ==============================================================================
@app.get(
    "/",
    tags=["Raiz"],
    summary="Página inicial informativa",
    description="Redireciona para a documentação interativa Swagger (/docs)."
)
async def root():
    """Redireciona automaticamente quem acessar a raiz para a documentação interativa."""
    return RedirectResponse(url="/docs", status_code=status.HTTP_307_TEMPORARY_REDIRECT)