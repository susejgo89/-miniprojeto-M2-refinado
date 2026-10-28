"""
Roteador para verificação de integridade operacional (Health Check).
Atende à Fase 5 do edital: GET /health retornando status HTTP 200 OK.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Request, status
from api.schemas.health import HealthCheckSchema

# Criamos o roteador com tag para agrupar na documentação Swagger
router = APIRouter(tags=["Monitoramento"])


@router.get(
    "/health",
    response_model=HealthCheckSchema,
    status_code=status.HTTP_200_OK,
    summary="Verifica a integridade da API e presença do modelo na RAM",
    description="Retorna se a API está ativa, o dispositivo de execução (CPU/CUDA) e confirmação de modelo carregado."
)
async def health_check(request: Request):
    """
    Endpoint que consulta o estado do modelo diretamente da memória RAM
    armazenado em app.state durante a inicialização (Lifespan).
    """
    # Recuperamos o serviço que foi guardado na memória RAM durante o Lifespan
    predictor_service = getattr(request.app.state, "predictor_service", None)

    modelo_carregado = predictor_service is not None and predictor_service.is_ready()
    versao = predictor_service.model_version if predictor_service else "desconhecido"
    dispositivo = predictor_service.device if predictor_service else "desconhecido"

    return HealthCheckSchema(
        status="healthy" if modelo_carregado else "degraded",
        model_loaded=modelo_carregado,
        model_version=versao,
        device=dispositivo,
        timestamp=datetime.now(timezone.utc).isoformat()
    )