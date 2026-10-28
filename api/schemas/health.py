"""
Módulo de Esquema Pydantic para Monitoramento de Saúde da API (/health).
Atende aos requisitos da Fase 4 e Fase 5.
"""

from pydantic import BaseModel, Field


class HealthCheckSchema(BaseModel):
    """
    Contrato de dados retornado pela rota GET /health.
    Informa a integridade operacional e confirma se o modelo está carregado na RAM.
    """
    status: str = Field(
        "healthy",
        description="Status operacional da API ('healthy' ou 'degraded')."
    )
    model_loaded: bool = Field(
        ...,
        description="Flag indicando se o artefato do modelo reside na memória RAM da aplicação."
    )
    model_version: str = Field(
        ...,
        description="Versão e identificador do modelo serializado em execução."
    )
    device: str = Field(
        ...,
        description="Dispositivo de execução computacional alocado para inferência (ex: 'CPU' ou 'CUDA')."
    )
    timestamp: str = Field(
        ...,
        description="Data e hora em formato ISO 8601 da realização do health check."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "MLP-Keras-v1.0",
                "device": "CPU",
                "timestamp": "2026-10-26T20:30:00Z"
            }
        }
    }