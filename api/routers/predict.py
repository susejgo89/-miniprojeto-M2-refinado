"""
Roteador para inferência de Machine Learning (MNIST).
Atende à Fase 5: Endpoints POST /predict e /v1/predict com validação estrita e tratamento de erros.
"""

from fastapi import APIRouter, Request, HTTPException, status
from api.schemas.prediction import MNISTInputSchema, MNISTOutputSchema

# Criamos o roteador agrupado sob a tag "Inferência Preditiva" no Swagger
router = APIRouter(tags=["Inferência Preditiva"])


@router.post(
    "/predict",
    response_model=MNISTOutputSchema,
    status_code=status.HTTP_200_OK,
    summary="Realiza inferência de dígito manuscrito (0 a 9)",
    description="Recebe vetor de 784 pixels normalizados, executa predição com a Rede Neural MLP em RAM e retorna o dígito inferido e probabilidades."
)
@router.post(
    "/v1/predict",
    response_model=MNISTOutputSchema,
    status_code=status.HTTP_200_OK,
    include_in_schema=False  # Alias para compatibilidade de versionamento
)
async def predict_digit(payload: MNISTInputSchema, request: Request):
    """
    Controlador HTTP de inferência:
    1. Recebe payload já rigorosamente validado pelo Pydantic (Fase 4).
    2. Invoca o serviço de predição em memória RAM desacoplado do HTTP (Fases 2 e 3).
    3. Trata exceções retornando status codes semânticos adequados (Fase 5).
    """
    # 1. Recuperamos o serviço que reside na memória RAM da aplicação
    predictor_service = getattr(request.app.state, "predictor_service", None)

    if predictor_service is None or not predictor_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O modelo preditivo ainda não está pronto ou não foi carregado na memória RAM."
        )

    # 2. Execução da inferência matemática com tratamento defensivo de erros
    try:
        classe_predita, confianca, probabilidades, tempo_ms = predictor_service.predict(
            payload.features
        )
    except ValueError as e:
        # Erro de negócio nos dados (ex: valores matematicamente inconsistentes)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dados de entrada inconsistentes para inferência: {str(e)}"
        )
    except Exception as e:
        # Falha imprevista no grafo de tensores: evita crash e retorna erro 500 controlado
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha interna durante a execução do modelo: {str(e)}"
        )

    # 3. Retornamos a resposta formatada estritamente segundo o MNISTOutputSchema
    return MNISTOutputSchema(
        predicted_class=classe_predita,
        confidence=confianca,
        probabilities=probabilidades,
        inference_time_ms=tempo_ms
    )